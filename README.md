# 🔥 Real-Time Fire and Smoke Detection System using YOLO

## 📘 Overview

This project is a real-time Fire and Smoke Detection System built using YOLO-based object detection models.

The system is designed to monitor live CCTV feeds, surveillance streams, and video footage to detect fire and smoke incidents in real time and trigger safety alerts for rapid response.

Developed during my internship, this solution is actively deployed in a live environment for continuous monitoring and fire safety operations.

---

# 🚀 Features

- 🔥 Real-time fire detection
- 💨 Smoke detection with bounding box localization
- 📹 Supports CCTV feeds, webcams, RTSP streams, and video files
- ⚡ Optimized YOLO inference for low latency
- 🧠 Custom-trained model on manually labeled datasets
- 🚨 Alert-ready architecture for emergency notification systems
- ☁️ Production deployment support
- 🧩 Modular architecture for scalability and integration

---

# 🧠 System Architecture

```text
Input Video Stream
        ↓
Frame Extraction
        ↓
YOLO Fire & Smoke Detection
        ↓
Bounding Box Prediction
        ↓
Alert Trigger / Visualization
        ↓
Real-Time Monitoring Dashboard
```

---

# 🧑‍💻 Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | YOLOv5 / YOLOv8 |
| Programming Language | Python |
| Computer Vision | OpenCV |
| Deep Learning Framework | PyTorch |
| Annotation Tool | Label Studio |
| Backend/API | Flask / FastAPI |
| Numerical Computing | NumPy |
| Deployment | GPU-enabled Server |

---

# 📂 Project Structure

```text
fire-smoke-detection-system/
│
├── app/
│   ├── live_detection.py
│   ├── inference.py
│   ├── alert_system.py
│   └── utils/
│
├── models/
│   └── fire_smoke_yolo.pt
│
├── notebooks/
│   └── fire_detection_training.ipynb
│
├── demo/
│   ├── demo_video.mp4
│   └── screenshots/
│
├── requirements.txt
├── README.md
└── app.py
```

---

# 📊 Model Capabilities

The YOLO model was custom-trained on fire and smoke datasets with manually annotated images to improve detection robustness under different environmental conditions such as:

- Indoor fire
- Outdoor fire
- Dense smoke
- Low-light environments
- Industrial monitoring scenarios

---

# 📸 Sample Results

## Fire Detection
(Add screenshot here)

## Smoke Detection
(Add screenshot here)

## Live Monitoring Output
(Add screenshot here)

---

# ⚙️ Installation

## Clone Repository

```bash
git clone https://github.com/your-username/fire-smoke-detection-system.git
cd fire-smoke-detection-system
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Detection System

```bash
python app/live_detection.py
```

---

# 🎥 Demo

(Add GIF or demo video here)

---

# 🌍 Real-World Applications

- Smart surveillance systems
- Industrial safety monitoring
- Warehouse fire prevention
- Forest fire early detection
- Smart city monitoring
- Public safety systems

---

# 🔮 Future Improvements

- SMS / Email alert integration
- Telegram / WhatsApp notifications
- Edge AI deployment
- Multi-camera monitoring support
- Cloud dashboard analytics
- Thermal camera integration

---

# 👨‍💻 Author

## Naman Jain

- GitHub: https://github.com/your-username
- LinkedIn: Add your LinkedIn profile

---

# ⭐ Acknowledgements

Special thanks to my internship team and the open-source computer vision community for supporting the development of this system.
