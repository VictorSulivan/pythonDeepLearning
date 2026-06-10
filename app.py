from fastapi import FastAPI, UploadFile, File
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import onnxruntime as ort
import numpy as np

app = FastAPI(title="CIFAR-10 ResNet18 API")

# Chargement du modèle ONNX pour l'inférence rapide
ort_session = ort.InferenceSession("resnet18_cifar10.onnx")

classes = ['avion', 'automobile', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'bateau', 'camion']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # Convertir en tableau numpy pour ONNX et ajouter la dimension batch
    return transform(image).unsqueeze(0).numpy()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_tensor = preprocess_image(image_bytes)
    
    # Inférence ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outs = ort_session.run(None, ort_inputs)
    
    # Récupération de la classe prédite
    logits = ort_outs[0]
    class_idx = int(np.argmax(logits, axis=1)[0])
    
    
    return {"class_index": class_idx, "label": classes[class_idx]}