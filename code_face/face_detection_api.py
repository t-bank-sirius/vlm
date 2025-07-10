from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import pickle
import numpy.linalg as LA
import base64
from io import BytesIO
from PIL import Image
import logging
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = 'buffalo_l'
DB_EMBEDDINGS_FILE = 'face_embeddings.npy'
LABELS_FILE = 'labels.pkl'
EMBEDDING_SIZE = 512
MAX_FACES = 4

class FaceRequest(BaseModel):
    image_base64: str

class AddFaceRequest(FaceRequest):
    comment: str

class CheckFaceResponse(BaseModel):
    found: bool
    name: str = None
    confidence: float = None
    message: str

class FaceRecognitionSystem:
    def __init__(self):
        self.face_app = FaceAnalysis(name=MODEL_NAME)
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.load_database()
    
    def load_database(self):
        if os.path.exists(DB_EMBEDDINGS_FILE) and os.path.exists(LABELS_FILE):
            self.face_embeddings = np.load(DB_EMBEDDINGS_FILE)
            with open(LABELS_FILE, 'rb') as f:
                self.face_labels = pickle.load(f)
        else:
            self.face_embeddings = np.empty((0, EMBEDDING_SIZE), dtype=np.float32)
            self.face_labels = []
    
    def save_database(self):
        np.save(DB_EMBEDDINGS_FILE, self.face_embeddings)
        with open(LABELS_FILE, 'wb') as f:
            pickle.dump(self.face_labels, f)
    
    def base64_to_image(self, base64_str):
        try:
            if 'base64,' in base64_str:
                base64_str = base64_str.split('base64,')[1]
            image_data = base64.b64decode(base64_str)
            image = Image.open(BytesIO(image_data))
            return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Ошибка декодирования base64: {str(e)}")
            return None
    
    def process_image(self, image):
        if image is None:
            return None, "Некорректное изображение"
        try:
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            faces = self.face_app.get(img_rgb)
            if len(faces) == 0:
                return None, "Лица не обнаружены"
            if len(faces) > MAX_FACES:
                return None, f"Слишком много лиц на фото ({len(faces)}). Максимально допустимо: {MAX_FACES}."
            main_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            return main_face.normed_embedding, None
        except Exception as e:
            logger.error(f"Ошибка обработки изображения: {str(e)}")
            return None, "Ошибка обработки изображения"
    
    def add_face(self, base64_str, comment):
        if not comment or comment.strip() == "":
            return "Комментарий не может быть пустым"
        
        image = self.base64_to_image(base64_str)
        if image is None:
            return "Невозможно декодировать изображение из base64"
        
        embedding, error = self.process_image(image)
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
                found=False,
                message="Невозможно декодировать изображение из base64"
            )
        
        embedding, error = self.process_image(image)
        if error:
            return CheckFaceResponse(
                found=False,
                message=error
            )
        
        if len(self.face_labels) == 0:
            return CheckFaceResponse(
                found=False,
                message="База данных лиц пуста"
            )
        
        dists = LA.norm(self.face_embeddings - np.array(embedding).reshape(1, -1), axis=1)
        min_index = np.argmin(dists)
        min_dist = dists[min_index]
        
        if min_dist < 0.6:
            confidence = max(0, 100 - min_dist * 100)
            return CheckFaceResponse(
                found=True,
                name=self.face_labels[min_index],
                confidence=confidence,
                message=f"Лицо найдено: {self.face_labels[min_index]} (уверенность: {confidence:.1f}%)"
            )
        else:
            return CheckFaceResponse(
                found=False,
                message="Совпадений в базе не найдено"
            )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

face_system = None

@app.on_event("startup")
async def startup_event():
    global face_system
    try:
        face_system = FaceRecognitionSystem()
    except Exception as e:
        logger.error(f"❌ Ошибка инициализации системы распознавания: {str(e)}")
        raise RuntimeError(f"Ошибка инициализации системы: {str(e)}")

@app.get("/health")
async def health_check():
    return {
        "status": "OK",
        "database_size": len(face_system.face_labels) if face_system else 0
    }

@app.post("/add_face")
async def add_face(request: AddFaceRequest):
    if not face_system:
        raise HTTPException(status_code=503, detail="Система распознавания не инициализирована")
    
    try:
        result = face_system.add_face(request.image_base64, request.comment)
        return {"result": result}
    except Exception as e:
        logger.error(f"Ошибка при добавлении лица: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

@app.post("/check", response_model=CheckFaceResponse)
async def check_face(request: FaceRequest):
    if not face_system:
        raise HTTPException(status_code=503, detail="Система распознавания не инициализирована")
    
    try:
        return face_system.check_face(request.image_base64)
    except Exception as e:
        logger.error(f"Ошибка при проверке лица: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)