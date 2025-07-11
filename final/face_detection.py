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

MODEL_NAME = 'buffalo_l'
DB_EMBEDDINGS_FILE = 'face_embeddings.npy'
LABELS_FILE = 'labels.pkl'
EMBEDDING_SIZE = 512
MAX_FACES = 1

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
        except:
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
        except:
            return None, "Ошибка обработки изображения"
    
    def add_face(self, base64_str, comment):
        if not comment or comment.strip() == "":
            return "Ошибка: Комментарий не может быть пустым"
        
        image = self.base64_to_image(base64_str)
        if image is None:
            return "Ошибка: Невозможно декодировать изображение из base64"
        
        embedding, error = self.process_image(image)
        if error:
            return f"Ошибка: {error}"
        
        if len(self.face_labels) > 0:
            dists = LA.norm(self.face_embeddings - np.array(embedding).reshape(1, -1), axis=1)
            min_dist = np.min(dists)
            if min_dist < 0.5:
                duplicate_index = np.argmin(dists)
                duplicate_name = self.face_labels[duplicate_index]
                return f"Предупреждение: Лицо уже существует в базе как '{duplicate_name}'"
        
        try:
            self.face_embeddings = np.vstack([self.face_embeddings, np.array(embedding).reshape(1, -1)])
            self.face_labels.append(comment)
            self.save_database()
            return f"Успех: Лицо '{comment}' добавлено в базу"
        except:
            return "Ошибка: Невозможно добавить лицо в базу"

def main():
    base64_str = input("Введите base64 изображения: ").strip()
    if not base64_str:
        print("Ошибка: Пустой ввод base64")
        return
    
    comment = input("Введите комментарий (кто на фото): ").strip()
    if not comment:
        print("Ошибка: Комментарий не может быть пустым")
        return
    
    try:
        frs = FaceRecognitionSystem()
    except:
        print("Ошибка: Невозможно инициализировать систему распознавания лиц")
        return
    
    result = frs.add_face(base64_str, comment)
    print(result)

if __name__ == "__main__":
    main()