from fastapi import FastAPI, HTTPException
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
from typing import List, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VLM_MODEL_PATH = os.getenv("VLM_MODEL_PATH", "Qwen/Qwen-VL-Chat-Int4")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_MODEL_NAME = 'buffalo_l'
DB_EMBEDDINGS_FILE = 'face_embeddings.npy'
LABELS_FILE = 'labels.pkl'
EMBEDDING_SIZE = 512
MAX_FACES_FOR_ADD = 1
MAX_FACES_FOR_CHECK = 8

class VLMGenerateRequest(BaseModel):
    image_base64: str
    prompt: str = None

class VLMGenerateResponse(BaseModel):
    description: str
    processing_time: float
    model_used: str

class FaceRequest(BaseModel):
    image_base64: str

class AddFaceRequest(FaceRequest):
    comment: str

class FaceMatchResult(BaseModel):
    bbox: List[float]
    found: bool
    name: Optional[str] = None
    confidence: Optional[float] = None
    message: str

class CheckFaceResponse(BaseModel):
    faces: List[FaceMatchResult]
    message: str

class HealthResponse(BaseModel):
    status: str
    vlm_loaded: bool
    face_recognition_loaded: bool
    device: str
    face_db_size: int

class FaceRecognitionSystem:
    def __init__(self):
        self.face_app = FaceAnalysis(name=FACE_MODEL_NAME)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.load_database()
    
    def load_database(self):
        if os.path.exists(DB_EMBEDDINGS_FILE) and os.path.exists(LABELS_FILE):
            self.face_embeddings = np.load(DB_EMBEDDINGS_FILE)
            with open(LABELS_FILE, 'rb') as f:
                self.face_labels = pickle.load(f)
            logger.info(f"База лиц загружена: {len(self.face_labels)} записей")
        else:
            self.face_embeddings = np.empty((0, EMBEDDING_SIZE), dtype=np.float32)
            self.face_labels = []
            logger.info("Создана новая база лиц")
    
    def save_database(self):
        np.save(DB_EMBEDDINGS_FILE, self.face_embeddings)
        with open(LABELS_FILE, 'wb') as f:
            pickle.dump(self.face_labels, f)
        logger.info("База лиц сохранена")
    
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
    
    def process_image_for_add(self, image):
        if image is None:
            return None, "Некорректное изображение"
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            if len(faces) == 0:
                return None, "Лица не обнаружены"
            if len(faces) > MAX_FACES_FOR_ADD:
                return None, "На фото больше одного лица. Пожалуйста, загрузите фото с одним лицом."
            
            main_face = faces[0]
            return main_face.normed_embedding, None
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {str(e)}")
            return None, "Ошибка обработки изображения"
    
    def process_image_for_check(self, image):
        if image is None:
            return [], "Некорректное изображение"
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            if len(faces) == 0:
                return [], "Лица не обнаружены"
            if len(faces) > MAX_FACES_FOR_CHECK:
                faces = faces[:MAX_FACES_FOR_CHECK]
            
            results = []
            for face in faces:
                bbox = face.bbox.tolist()
                embedding = face.normed_embedding
                results.append((bbox, embedding))
            
            return results, None
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {str(e)}")
            return [], "Ошибка обработки изображения"
    
    def add_face(self, base64_str, comment):
        if not comment or comment.strip() == "":
            return "Комментарий не может быть пустым"
        
        image = self.base64_to_image(base64_str)
        if image is None:
            return "Невозможно декодировать изображение из base64"
        
        embedding, error = self.process_image_for_add(image)
        if error:
            return error
        
        if len(self.face_labels) > 0:
            dists = LA.norm(self.face_embeddings - np.array(embedding).reshape(1, -1), axis=1)
            min_dist = np.min(dists)
            if min_dist < 0.5:
                duplicate_index = np.argmin(dists)
                duplicate_name = self.face_labels[duplicate_index]
                return f"Лицо уже существует в базе как '{duplicate_name}'"
        
        try:
            self.face_embeddings = np.vstack([self.face_embeddings, np.array(embedding).reshape(1, -1)])
            self.face_labels.append(comment)
            self.save_database()
            return f"Лицо '{comment}' успешно добавлено в базу"
        except Exception as e:
            logger.error(f"Ошибка добавления лица: {str(e)}")
            return "Невозможно добавить лицо в базу"
    
    def check_face(self, base64_str):
        image = self.base64_to_image(base64_str)
        if image is None:
            return CheckFaceResponse(
                faces=[],
                message="Невозможно декодировать изображение из base64"
            )
        
        faces, error = self.process_image_for_check(image)
        if error:
            return CheckFaceResponse(
                faces=[],
                message=error
            )
        
        if len(self.face_labels) == 0:
            return CheckFaceResponse(
                faces=[],
                message="База данных лиц пуста"
            )
        
        results = []
        for bbox, embedding in faces:
            dists = LA.norm(self.face_embeddings - np.array(embedding).reshape(1, -1), axis=1)
            min_index = np.argmin(dists)
            min_dist = dists[min_index]
            
            if min_dist < 0.6:
                confidence = max(0, 100 - min_dist * 100)
                results.append(FaceMatchResult(
                    bbox=bbox,
                    found=True,
                    name=self.face_labels[min_index],
                    confidence=confidence,
                    message=f"Найдено: {self.face_labels[min_index]} ({confidence:.1f}%)"
                ))
            else:
                results.append(FaceMatchResult(
                    bbox=bbox,
                    found=False,
                    message="Не найдено в базе"
                ))
        
        return CheckFaceResponse(
            faces=results,
            message=f"Обработано {len(results)} лиц"
        )

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
    title="Multimodal API",
    description="API для генерации описаний изображений и распознавания лиц",
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
    "Describe what you see in the photo. Speak like with a person from 8 to 14 years old. "
    "Speak only English. In math tasks, 'x' is a mathematical variable, not multiplication. "
    "Example: for 6x + 5 = 23, it's 6x = 23-5 → 6x = 18 → x = 3"
)

@app.get("/health", response_model=HealthResponse)
async def health_check():
    face_db_size = len(app.state.face_system.face_labels) if app.state.face_system else 0
    return HealthResponse(
        status="healthy",
        vlm_loaded=app.state.vlm_loaded,
        face_recognition_loaded=app.state.face_recognition_loaded,
        device=DEVICE,
        face_db_size=face_db_size
    )

@app.post("/vlm/describe", response_model=VLMGenerateResponse)
async def describe_image(request: VLMGenerateRequest):
    start_time = time.time()
    
    if not hasattr(app.state, "model") or not app.state.vlm_loaded:
        raise HTTPException(status_code=503, detail="VLM модель не загружена")
    
    tmp_path = None
    try:
        base64_str = request.image_base64
        prompt = request.prompt if request.prompt else DEFAULT_PROMPT
        
        if ";base64," in base64_str:
            base64_str = base64_str.split(";base64,")[1]
        
        image_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(image_data))
        
        if img.mode != "RGB":
            img = img.convert("RGB")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp, format="JPEG")
            tmp_path = tmp.name
        
        query = app.state.tokenizer.from_list_format([
            {'image': tmp_path},
            {'text': prompt},
        ])
        
        with torch.no_grad():
            response, _ = app.state.model.chat(
                tokenizer=app.state.tokenizer,
                query=query,
                history=None
            )
        
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Описание сгенерировано за {processing_time:.2f}с")
        
        return VLMGenerateResponse(
            description=response,
            processing_time=processing_time,
            model_used=VLM_MODEL_PATH
        )
    
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)
        logger.error(f"❌ Ошибка обработки изображения: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")

@app.post("/face/add")
async def add_face(request: AddFaceRequest):
    if not app.state.face_system or not app.state.face_recognition_loaded:
        raise HTTPException(status_code=503, detail="Система распознавания лиц не инициализирована")
    
    try:
        result = app.state.face_system.add_face(request.image_base64, request.comment)
        return {"result": result}
    except Exception as e:
        logger.error(f"Ошибка при добавлении лица: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/face/check", response_model=CheckFaceResponse)
async def check_face(request: FaceRequest):
    if not app.state.face_system or not app.state.face_recognition_loaded:
        raise HTTPException(status_code=503, detail="Система распознавания лиц не инициализирована")
    
    try:
        return app.state.face_system.check_face(request.image_base64)
    except Exception as e:
        logger.error(f"Ошибка при проверке лица: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)