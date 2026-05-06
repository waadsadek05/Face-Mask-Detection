# 😷 Face Mask Detection

> MobileNetV2 · FastAPI · Docker · OpenCV Face Detection

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Transfer%20Learning-EE4C2C?style=flat-square&logo=pytorch)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square&logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker)
![Accuracy](https://img.shields.io/badge/Target%20Accuracy-%3E90%25-success?style=flat-square)

---

## 📌 Project Overview

Detect whether a person is wearing a face mask using a **MobileNetV2** model trained on 12,000 real-world images.

**Key feature:** An OpenCV face detector crops the face region *before* inference — so the model works correctly on full-body photos, not just close-up face shots.

| Item | Details |
|------|---------|
| Dataset | [Face Mask 12K Images — Kaggle](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset) |
| Classes | `WithMask` / `WithoutMask` |
| Total Images | 12,000 |
| Model | MobileNetV2 (transfer learning) |
| Export | `mask_detector.pth` |
| Target | > 90% accuracy |

---

## 🛠️ Technology Stack

| Technology | Purpose |
|------------|---------|
| PyTorch + TorchVision | Training and transfer learning |
| FastAPI + Uvicorn | REST API deployment |
| Docker | Containerized deployment |
| Pillow | Image preprocessing |
| scikit-learn | Confusion matrix, classification report |
| OpenCV | Face detection & cropping before inference |

---

## 📁 Project Structure

```
FaceMaskDetection/
├── api/
│   ├── app.py                  # FastAPI app with face detection
│   ├── static/
│   │   └── index.html          # Frontend UI (dark/light mode)
│   └── Dockerfile
├── model/
│   └── mask_detector.pth       # Trained MobileNetV2 weights
├── notebooks/
│   ├── 1_data_eda.ipynb
│   ├── 2_eda_augmentation.ipynb
│   ├── 3_augmentation.ipynb
│   ├── 4_model_training.ipynb
│   └── 5_evaluation.ipynb
└── requirements.txt
```

---

## 👩‍💻 Team Roles

| # | Role | Tasks |
|---|------|-------|
| 1 | Data Manager | Download dataset, organize `with_mask` / `without_mask` folders, verify quality |
| 2 | EDA & Visualizer | Sample images per class, face diversity analysis, statistics |
| 3 | Augmentation | Horizontal flip, brightness, rotation, noise pipeline |
| 4 | Model Trainer | MobileNetV2 transfer learning, training curves, early stopping |
| 5 | Evaluator | Confusion matrix, per-class metrics, test with diverse faces |
| 6 | API Developer | FastAPI `/predict` with status + confidence + recommended action |
| 7 | Deployer | Docker, test with webcam/phone photos, demo, presentation |

---

## 📅 Timeline

| Week | Milestone | Responsible |
|------|-----------|-------------|
| Week 1 | Dataset download, EDA notebook, augmentation | Roles 1–3 |
| Week 2 | Train model, export `.pth`, evaluate | Roles 3–5 |
| Week 3 | Build FastAPI, Dockerfile, test script | Roles 5–7 |
| Week 4 | Integration test, presentation prep, live demo | All |

---

## 🤖 Face Detection Pipeline

The model was trained on cropped face images — so sending a full-body photo without preprocessing causes wrong predictions. We solve this with an OpenCV step before every inference:

```
Input Image
    ↓
OpenCV Haar Cascade → detect face
    ↓
Crop face + 20% margin
    ↓
MobileNetV2 inference
    ↓
Result ✅

(If no face detected → fallback to full image)
```

---

## 🚀 Run with Docker

```bash
# Build
docker build -t mask-detector -f api/Dockerfile .

# Run
docker run -p 8000:8000 mask-detector
```

Then open your browser:

- **UI** → http://localhost:8000
- **Swagger Docs** → http://localhost:8000/docs

---

## 💻 Run Locally (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Start the API
cd api
uvicorn app:app --reload --port 8000
```

---

## 📡 API Reference

### `POST /predict`

Upload an image to get a mask detection result.

**Request:** `multipart/form-data` with field `file` (jpg / png / webp)

**Response:**

```json
{
  "status":        "mask_on",
  "action":        "Allow entry",
  "confidence":    0.96,
  "face_detected": true
}
```

| Field | Type | Values |
|-------|------|--------|
| `status` | string | `mask_on` / `mask_off` |
| `action` | string | `Allow entry` / `Deny entry` |
| `confidence` | float | `0.0` – `1.0` |
| `face_detected` | boolean | `true` / `false` |

---

## 📦 Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | EDA Notebook | 8+ visualizations with written insights |
| 2 | Trained Model | Exported as `mask_detector.pth` |
| 3 | Training Report | Loss/accuracy curves, best epoch, final metrics |
| 4 | Evaluation | Confusion matrix, per-class accuracy, F1 scores |
| 5 | FastAPI App | `/predict` endpoint accepting image uploads |
| 6 | Dockerfile | Containerized deployment ready |
| 7 | Test Script | Automated API testing — `test_api.py` |
| 8 | Presentation | 10-minute demo with live API test |

---

## ⚠️ Key Challenges

- **Diverse faces** — model must generalize across demographics
- **Lighting variation** — indoor vs. outdoor conditions
- **Mask types** — surgical, cloth, N95 all look different
- **Full-body photos** — solved by OpenCV face cropping

## 💡 Tips

- Use **horizontal flip** augmentation — do NOT use vertical flip
- Apply **brightness augmentation** for indoor/outdoor lighting
- Check `face_detected` field in the response to verify cropping is working
- Test with team member photos for real-world validation


---

## 👥 Team — Face Mask Detection

| # | Name |
|---|------|
| 1 | Waad |
| 2 | Aliaa |
| 3 | Alaa |
| 4 | Sama |
| 5 | Noha |
| 6 | Hania |

---

<div align="center">
  <sub>Face Mask Detection · MobileNetV2 · Advanced ML Team Project</sub>
</div>