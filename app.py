from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware  # <-- IMPORT CRITIQUE
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="CIFAR-10 ResNet18 API")

# --- CONFIGURATION DU CORS ---
# On indique à FastAPI d'accepter les requêtes venant d'autres origines (comme ton serveur Live Server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permet à n'importe quel site (dont localhost:5500) d'appeler l'API
    allow_credentials=True,
    allow_methods=["*"],  # Permet le POST, GET, etc.
    allow_headers=["*"],  # Permet tous les headers
)

# Chargement du modèle ONNX
ort_session = ort.InferenceSession("resnet18_cifar10.onnx")
input_name = ort_session.get_inputs()[0].name

CLASSES = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((32, 32)) # (ou 224, 224 si tu as fait le Resize à l'entraînement)
    img_np = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
    std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32)
    img_np = (img_np - mean) / std
    img_np = img_np.transpose(2, 0, 1)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image(image_bytes)
        
        ort_inputs = {input_name: input_tensor}
        ort_outs = ort_session.run(None, ort_inputs)
        logits = ort_outs[0]
        
        class_idx = int(np.argmax(logits, axis=1)[0])
        score = float(np.max(logits, axis=1)[0])
        
        return {
            "success": True,
            "class_index": class_idx,
            "label": CLASSES[class_idx],
            "confidence_logit": round(score, 4)
        }
    except Exception as e:
        return {"success": False, "error": str(e)}