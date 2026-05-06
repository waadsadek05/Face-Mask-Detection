from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, UnidentifiedImageError
import io
import cv2
import numpy as np
import os

app = FastAPI(title="Face Mask Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Classes ──────────────────────────────────
CLASSES = ['with_mask', 'without_mask']

# ── Load Mask Detection Model ─────────────────
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASSES))
model.load_state_dict(torch.load("model/mask_detector.pth", map_location='cpu'))
model.eval()

# ── Load OpenCV Face Detector (Haar Cascade) ──
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ── Preprocessing ─────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ── Face Detection ────────────────────────────
def detect_and_crop_face(pil_image: Image.Image):
    img_np = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return pil_image, False

    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, w, h = faces[0]

    margin_x = int(w * 0.20)
    margin_y = int(h * 0.20)

    img_w, img_h = pil_image.size
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    face_crop = pil_image.crop((x1, y1, x2, y2))
    return face_crop, True


# ── Helper ────────────────────────────────────
def map_result(pred_class: str):
    if pred_class == 'with_mask':
        return "mask_on", "Allow entry"
    return "mask_off", "Deny entry"


# ── Serve Frontend ────────────────────────────
@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# ── Predict ───────────────────────────────────
@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=422, detail="File must be an image (jpg, png, …)")

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, Exception):
        raise HTTPException(status_code=422, detail="Could not open file as an image.")

    face_image, face_found = detect_and_crop_face(image)

    tensor = transform(face_image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    pred_idx   = probs.argmax().item()
    pred_class = CLASSES[pred_idx]
    confidence = float(probs.max())
    status, action = map_result(pred_class)

    return {
        "status":        status,
        "action":        action,
        "confidence":    round(confidence, 2),
        "face_detected": face_found
    }