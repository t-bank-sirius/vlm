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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

VLM_MODEL_PATH = os.getenv("VLM_MODEL_PATH", "Qwen/Qwen-VL-Chat-Int4")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class VLMGenerateRequest(BaseModel):
    image_base64: str
    prompt: str = None

class VLMGenerateResponse(BaseModel):
    description: str
    processing_time: float
    model_used: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Запуск VLM API...")
    
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
        logger.info(f"✅ Модель загружена на {DEVICE}")
        app.state.model_loaded = True
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {str(e)}")
        app.state.model_loaded = False
    
    yield
    
    logger.info("🛑 Остановка VLM API...")

app = FastAPI(
    title="VLM API",
    description="Vision-Language Model API",
    version="0.1.0",
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
    return HealthResponse(
        status="healthy" if hasattr(app.state, "model_loaded") and app.state.model_loaded else "degraded",
        model_loaded=app.state.model_loaded,
        device=DEVICE
    )

@app.post("/v1/describe", response_model=VLMGenerateResponse)
async def describe_image(request: VLMGenerateRequest):
    start_time = time.time()
    
    if not hasattr(app.state, "model") or not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Модель не загружена")
    
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8880)