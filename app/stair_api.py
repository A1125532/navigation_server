from fastapi import APIRouter, UploadFile, File
import onnxruntime as ort
import numpy as np
from PIL import Image
import io
import os

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "data", "model", "segformer_stair.onnx")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

def preprocess(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((512, 512))

    arr = np.array(image).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)

@router.post("/detect/stair")
async def detect_stair(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess(image_bytes)

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_tensor})

    result = outputs[0]

    # Segmentation 常見輸出格式: [1, class_count, H, W]
    pred = np.argmax(result, axis=1)[0]

    # 這裡 class id 先暫時用 1
    # 之後要依照你的模型實際樓梯 class id 修改
    stair_mask = (pred == 59)

    stair_ratio = float(np.sum(stair_mask) / stair_mask.size)
    has_stair = stair_ratio > 0.03

    return {
        "has_stair": has_stair,
        "stair_ratio": stair_ratio,
        "message": "前方有樓梯" if has_stair else "未偵測到樓梯"
    }