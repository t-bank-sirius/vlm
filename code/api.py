from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
import os
import base64
import io
import tempfile
from contextlib import asynccontextmanager
from pydantic import BaseModel
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import cv2
import numpy as np
from insightface.app import FaceAnalysis
import pickle
import numpy.linalg as LA
import httpx

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VLM_MODEL_PATH = os.getenv("VLM_MODEL_PATH", "/data/model")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INSIGHTFACE_ROOT = os.getenv("INSIGHTFACE_ROOT", "/data/.insightface")
FACE_MODEL_NAME = 'buffalo_l'
# DB_EMBEDDINGS_FILE = 'face_embeddings.npy'
# LABELS_FILE = 'labels.pkl'
USER_DATA_BASE_DIR = "user_data"
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", 512))
MAX_FACES_FOR_ADD = int(os.getenv("MAX_FACES_FOR_ADD", 1))
MAX_FACES_FOR_CHECK = int(os.getenv("MAX_FACES_FOR_CHECK", 8))

logger.info(DEVICE)


class FaceAddRequest(BaseModel):
    image_base64: str
    comment: str
    user_id: str

class AnalyzeRequest(BaseModel):
    image_base64: str
    prompt: str
    user_id: str

class SafetyRequest(BaseModel):
    image_base64: str
    user_id: str

class FaceAddResponse(BaseModel):
    result: str

class AnalyzeResponse(BaseModel):
    result: str
    processing_time: float

class SafetyResponse(BaseModel):
    result: str
    processing_time: float

class FaceRecognitionSystem:
    def __init__(self):
        self.face_app = FaceAnalysis(name=FACE_MODEL_NAME, root=INSIGHTFACE_ROOT)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        # self.load_database() # Database will be loaded on demand per user
    
    def _get_user_data_paths(self, user_id):
        user_dir = os.path.join(USER_DATA_BASE_DIR, user_id)
        os.makedirs(user_dir, exist_ok=True)
        db_path = os.path.join(user_dir, "face_embeddings.npy")
        labels_path = os.path.join(user_dir, "labels.pkl")
        return db_path, labels_path

    def load_database(self, user_id):
        db_path, labels_path = self._get_user_data_paths(user_id)
        if os.path.exists(db_path) and os.path.exists(labels_path):
            face_embeddings = np.load(db_path)
            with open(labels_path, 'rb') as f:
                face_labels = pickle.load(f)
            logger.info(f"База лиц для пользователя {user_id} загружена: {len(face_labels)} записей")
            return face_embeddings, face_labels
        else:
            logger.info(f"Создана новая база лиц для пользователя {user_id}")
            return np.empty((0, EMBEDDING_SIZE), dtype=np.float32), []
    
    def save_database(self, user_id, face_embeddings, face_labels):
        db_path, labels_path = self._get_user_data_paths(user_id)
        np.save(db_path, face_embeddings)
        with open(labels_path, 'wb') as f:
            pickle.dump(face_labels, f)
        logger.info(f"База лиц для пользователя {user_id} сохранена")
    
    def base64_to_image(self, base64_str):
        try:
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Ошибка декодирования base64: {str(e)}")
            return None
    
    def detect_main_face(self, image):
        if image is None:
            return None, "Некорректное изображение"
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            
            if len(faces) == 0:
                return None, "Лица не обнаружены"
            
            main_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            return main_face.normed_embedding, None
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {str(e)}")
            return None, "Ошибка обработки изображения"
    
    def detect_all_faces(self, image):
        if image is None:
            return [], "Некорректное изображение"
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            
            if len(faces) == 0:
                return [], "Лица не обнаружены"
            
            if len(faces) > MAX_FACES_FOR_CHECK:
                faces = faces[:MAX_FACES_FOR_CHECK]
            
            embeddings = [face.normed_embedding for face in faces]
            return embeddings, None
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {str(e)}")
            return [], "Ошибка обработки изображения"
    
    async def add_face(self, user_id, base64_str, comment):
        if not comment or comment.strip() == "":
            return "Комментарий не может быть пустым"
        
        face_embeddings, face_labels = self.load_database(user_id)

        image = self.base64_to_image(base64_str)
        if image is None:
            return "Невозможно декодировать изображение из base64"
        
        embedding, error = self.detect_main_face(image)
        if error:
            return error
        
        if len(face_labels) > 0:
            if len(face_labels) != len(face_embeddings):
                logger.error(
                    f"Несоответствие базы для пользователя {user_id}: метки={len(face_labels)}, эмбеддинги={len(face_embeddings)}"
                )
                min_length = min(len(face_labels), len(face_embeddings))
                face_labels = face_labels[:min_length]
                face_embeddings = face_embeddings[:min_length]
                self.save_database(user_id, face_embeddings, face_labels)
                logger.warning(f"База для пользователя {user_id} автоматически исправлена до {min_length} записей")
            
            dists = LA.norm(face_embeddings - np.array(embedding).reshape(1, -1), axis=1)
            min_dist = np.min(dists)
            if min_dist < 0.5:
                duplicate_index = np.argmin(dists)
                
                if duplicate_index < len(face_labels):
                    duplicate_name = face_labels[duplicate_index]
                    return f"Человек уже существует в базе как '{duplicate_name}'"
                else:
                    logger.error(
                        f"Ошибка индекса для пользователя {user_id}: {duplicate_index} (размер меток: {len(face_labels)})"
                    )
                    return "Ошибка: несоответствие данных в базе лиц"
        
        try:
            face_embeddings = np.vstack([face_embeddings, np.array(embedding).reshape(1, -1)])
            face_labels.append(comment)
            self.save_database(user_id, face_embeddings, face_labels)
            local_result = f"Ok! I'll remember that it's {comment}"
        except Exception as e:
            logger.error(f"Ошибка добавления лица для пользователя {user_id}: {str(e)}")
            local_result = "Невозможно добавить лицо в базу"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://ltm-api:8006/add_face",
                    json={
                        "user_id": user_id,
                        "name": comment,
                        "image": base64_str
                    },
                    timeout=30.0
                )
                response.raise_for_status()
                logger.info(f"Запрос к LTM API успешен: {response.json()}")
                return local_result
        except httpx.RequestError as e:
            logger.error(f"Ошибка запроса к LTM API: {e}")
            return f"{local_result}. Ошибка при синхронизации с LTM API: {e}"
        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка HTTP статуса от LTM API: {e.response.status_code} - {e.response.text}")
            return f"{local_result}. Ошибка HTTP от LTM API: {e.response.status_code}"
        except Exception as e:
            logger.error(f"Неизвестная ошибка при вызове LTM API: {e}")
            return f"{local_result}. Неизвестная ошибка при синхронизации с LTM API: {e}"
    
    def recognize_faces(self, user_id, base64_str):
        image = self.base64_to_image(base64_str)
        if image is None:
            return [], "Невозможно декодировать изображение из base64"
        
        embeddings, error = self.detect_all_faces(image)
        if error:
            return [], error
        
        face_embeddings, face_labels = self.load_database(user_id)

        if len(face_labels) == 0:
            return [], "База данных лиц пуста"
        
        results = []
        for embedding in embeddings:
            dists = LA.norm(face_embeddings - np.array(embedding).reshape(1, -1), axis=1)
            min_index = np.argmin(dists)
            min_dist = dists[min_index]
            
            if min_dist < 0.7:
                confidence = max(0, 100 - min_dist * 100)
                results.append({
                    "name": face_labels[min_index],
                    "confidence": confidence
                })
        
        return results, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск API...")
    
    try:
        app.state.tokenizer = AutoTokenizer.from_pretrained(
            VLM_MODEL_PATH,
            trust_remote_code=True
        )
        app.state.model = AutoModelForCausalLM.from_pretrained(
            VLM_MODEL_PATH,
            device_map=DEVICE,
            trust_remote_code=True
        ).eval()
        logger.info(f"✅ VLM модель загружена на {DEVICE}")
        app.state.vlm_loaded = True
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки VLM модели: {str(e)}")
        app.state.vlm_loaded = False
    
    try:
        app.state.face_system = FaceRecognitionSystem()
        logger.info("✅ Система распознавания лиц инициализирована")
        app.state.face_recognition_loaded = True
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации системы распознавания лиц: {str(e)}")
        app.state.face_recognition_loaded = False
        app.state.face_system = None
    
    yield
    
    logger.info("🛑 Остановка API...")

app = FastAPI(
    title="Multimodal Analysis API",
    description="API для анализа изображений с поддержкой запоминания лиц, генерации описаний и проверки безопасности",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_PROMPT = (
    """
    You are a Vision-Language AI assistant with responsible safety protocols.

    Your task is to analyze images and provide detailed, neutral descriptions only if the image is clearly safe and appropriate.

    You MUST NOT analyze, describe, or transcribe any images that contain any of the following with high certainty:
    - Realistic or graphic violence (e.g. visible blood, severe wounds, corpses, mutilation, real body parts).
    - Sexual or explicit content (e.g. pornography, visible nudity, sexual activity).
    - Abuse, torture, rape, or physical harm to real people or animals.
    - Clearly illegal or deeply disturbing content (e.g. executions, terrorism, suicide, racism).

    IMPORTANT:
    Only block the response if you are confident the content falls into the above categories.

    If you are uncertain — for example, if the image is ambiguous, abstract, fictional, artistic, or unclear — proceed with a cautious and neutral description.

    If the content appears clearly unsafe, respond with:
    "This image contains potentially unsafe or disturbing content and cannot be described."

    You ARE allowed to analyze:
    - Drawings, cartoons, anime, video game or fantasy art.
    - Memes, stylized or abstract art.
    - Fictional violence, stylized gore, or harmless special effects, if not realistic.

    Always distinguish between real and fictional content. Do not assume content is real without strong visual evidence.
    """
)

SAFETY_PROMPT = """
You are a strict content safety analyzer. Your task is to detect any illegal, prohibited, or dangerous content in the image.

Analyze the image for the following content:
1. Graphic violence, gore, or real blood
2. Nudity, sexual content, or pornography
3. Illegal activities (drugs, weapons, terrorist symbols)
4. Child exploitation content
5. Hate speech symbols (racist, extremist)
6. Non-consensual intimate content
7. Self-harm or suicide content

Rules:
- Respond only with "SAFE" if image contains NONE of the above
- Respond with "UNSAFE: <reason>" if ANY prohibited content is detected
- Be specific about the reason when marking as unsafe
- Consider even partial matches and subtle representations
- Better mark as unsafe if uncertain
"""

async def generate_description(image_base64: str, prompt: str) -> str:
    start_time = time.time()
    
    if not hasattr(app.state, "model") or not app.state.vlm_loaded:
        return "VLM модель не загружена"
    
    tmp_path = None
    try:
        if ';base64,' in image_base64:
            image_base64 = image_base64.split(';base64,')[1]
        
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp, format="JPEG")
            tmp_path = tmp.name
        
        final_prompt = DEFAULT_PROMPT
        if prompt and prompt.strip():
            final_prompt = DEFAULT_PROMPT + " " + prompt.strip()
        
        query = app.state.tokenizer.from_list_format([
            {'image': tmp_path},
            {'text': final_prompt},
        ])
        
        with torch.no_grad():
            response, _ = app.state.model.chat(
                tokenizer=app.state.tokenizer,
                query=query,
                history=None
            )
        
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        logger.info(f"✅ Описание сгенерировано за {time.time() - start_time:.2f}с")
        return response
    
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"❌ Ошибка генерации описания: {str(e)}")
        return f"Ошибка генерации описания: {str(e)}"

async def check_safety(image_base64: str) -> str:
    start_time = time.time()
    
    if not hasattr(app.state, "model") or not app.state.vlm_loaded:
        return "UNSAFE: Safety model not loaded"
    
    tmp_path = None
    try:
        if ';base64,' in image_base64:
            image_base64 = image_base64.split(';base64,')[1]
        
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp, format="JPEG")
            tmp_path = tmp.name
        
        query = app.state.tokenizer.from_list_format([
            {'image': tmp_path},
            {'text': SAFETY_PROMPT},
        ])
        
        with torch.no_grad():
            response, _ = app.state.model.chat(
                tokenizer=app.state.tokenizer,
                query=query,
                history=None
            )
        
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        logger.info(f"✅ Safety check completed in {time.time() - start_time:.2f}s")
        return response.strip()
    
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"❌ Safety check error: {str(e)}")
        return f"UNSAFE: Safety check error - {str(e)}"

@app.post("/add", response_model=FaceAddResponse)
async def add_face_endpoint(request: FaceAddRequest):
    start_time = time.time()
    
    if not app.state.face_recognition_loaded or not app.state.face_system:
        return FaceAddResponse(result="Система распознавания лиц не инициализирована")
    
    result = await app.state.face_system.add_face(request.user_id, request.image_base64, request.comment)
    
    return FaceAddResponse(
        result=result,
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_image(request: AnalyzeRequest):
    start_time = time.time()
    
    face_detection_result = "Лиц не обнаружено"
    recognized_faces = []
    
    if app.state.face_recognition_loaded and app.state.face_system:
        image = app.state.face_system.base64_to_image(request.image_base64)
        if image is not None:
            embeddings, error = app.state.face_system.detect_all_faces(image)
            
            if embeddings:
                face_detection_result = f"Обнаружено лиц: {len(embeddings)}"
                
                recognized, _ = app.state.face_system.recognize_faces(request.user_id, request.image_base64)
                if recognized:
                    recognized_faces = [f"{face['name']}" for face in recognized]
    
    if "не обнаружено" in face_detection_result.lower():
        description = await generate_description(request.image_base64, request.prompt)
        return AnalyzeResponse(
            result=description,
            processing_time=time.time() - start_time
        )
    
    if recognized_faces:
        identity_msg = "I remember this person! This is " + ", ".join(recognized_faces) + ". "
        
        description = await generate_description(request.image_base64, request.prompt)
        
        return AnalyzeResponse(
            result=identity_msg + description,
            processing_time=time.time() - start_time
        )
    else:
        description = await generate_description(request.image_base64, request.prompt)
        return AnalyzeResponse(
            result=description,
            processing_time=time.time() - start_time
        )

@app.post("/safety", response_model=SafetyResponse)
async def safety_check(request: SafetyRequest):
    start_time = time.time()
    result = await check_safety(request.image_base64)
    return SafetyResponse(
        result=result,
        processing_time=time.time() - start_time
    )
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
