import streamlit as st
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import faiss
import os
import pickle
from collections import Counter

MODEL_NAME = 'buffalo_l'
DB_INDEX_FILE = 'face_index.faiss'
LABELS_FILE = 'labels.pkl'
EMBEDDING_SIZE = 512
MAX_FACES = 4

@st.cache_resource
def load_model():
    app = FaceAnalysis(name=MODEL_NAME)
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app

face_app = load_model()

if os.path.exists(DB_INDEX_FILE):
    face_index = faiss.read_index(DB_INDEX_FILE)
    with open(LABELS_FILE, 'rb') as f:
        face_labels = pickle.load(f)
else:
    face_index = faiss.IndexFlatL2(EMBEDDING_SIZE)
    face_labels = []

def process_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = face_app.get(img_rgb)
    
    if len(faces) == 0:
        return None, "Лица не обнаружены"
    
    if len(faces) > MAX_FACES:
        return None, f"Слишком много лиц на фото ({len(faces)}). Максимально допустимо: {MAX_FACES}."
    
    main_face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
    return main_face.normed_embedding, None

def show_database_stats():
    if face_labels:
        st.sidebar.subheader("Статистика базы данных")
        counter = Counter(face_labels)
        for name, count in counter.items():
            st.sidebar.info(f"{name}: {count} {'лицо' if count == 1 else 'лица'}")
        st.sidebar.write(f"Всего лиц: {len(face_labels)}")

st.title("👨‍👩‍👧‍👦 Система распознавания лиц")

show_database_stats()

st.subheader("➕ Добавить новое лицо в базу")
with st.expander("Форма добавления", expanded=True):
    uploaded_file = st.file_uploader("Загрузите фото человека", type=['jpg', 'png', 'jpeg'], key="add")
    person_name = st.text_input("Кто это на фото? (Например: Моя мама)", key="name")

    if uploaded_file is not None and person_name:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.image(image, channels="BGR", caption="Загруженное изображение", width=300)
        
        embedding, error = process_image(image)
        
        if error:
            st.error(error)
        else:
            if face_labels:
                distances, _ = face_index.search(np.array([embedding]), 1)
                if distances[0][0] < 0.5:
                    st.warning("⚠️ Это лицо уже существует в базе данных!")
            
            if st.button("Добавить в базу"):
                face_index.add(np.array([embedding]))
                face_labels.append(person_name)
                
                faiss.write_index(face_index, DB_INDEX_FILE)
                with open(LABELS_FILE, 'wb') as f:
                    pickle.dump(face_labels, f)
                
                st.success(f"✅ Лицо '{person_name}' успешно добавлено в базу!")
                st.balloons()
                show_database_stats()

st.subheader("🔍 Поиск лиц в базе данных")
with st.expander("Форма поиска", expanded=True):
    search_file = st.file_uploader("Загрузите фото для поиска", type=['jpg', 'png', 'jpeg'], key="search")

    if search_file:
        file_bytes = np.asarray(bytearray(search_file.read()), dtype=np.uint8)
        search_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        st.image(search_image, channels="BGR", caption="Изображение для поиска", width=300)
        
        embedding, error = process_image(search_image)
        
        if error:
            st.error(error)
        else:
            distances, indices = face_index.search(np.array([embedding]), 3)
            
            found = False
            for i in range(len(distances[0])):
                if distances[0][i] < 0.6:
                    match = face_labels[indices[0][i]]
                    confidence = max(0, 100 - distances[0][i] * 100)
                    st.success(f"**Найдено совпадение:** {match} (Уверенность: {confidence:.1f}%)")
                    found = True
                    break
            
            if not found:
                st.warning("❌ Совпадений в базе не найдено")
                
            if found and face_labels:
                st.subheader("Топ-3 возможных совпадения:")
                cols = st.columns(3)
                for i, col in enumerate(cols):
                    if i < len(distances[0]):
                        idx = indices[0][i]
                        col.metric(label=face_labels[idx], value=f"{max(0, 100 - distances[0][i] * 100):.1f}%")
                        col.progress(min(100, int(100 - distances[0][i] * 100)))
