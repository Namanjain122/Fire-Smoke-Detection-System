<img width="300" height="168" alt="download" src="https://github.com/user-attachments/assets/cd3f9093-447b-431d-b28b-ff993e864116" />
# 🔥 Real-Time Fire & Smoke Detection System using YOLO

<div align="center">

### Intelligent Real-Time Surveillance for Fire Safety Monitoring

AI-powered fire and smoke detection system built with YOLO, OpenCV, and Flask for real-time CCTV surveillance and emergency monitoring.

</div>

---

# 📌 Overview

This project is a **real-time Fire and Smoke Detection System** developed using **YOLO-based object detection models** for intelligent surveillance and safety monitoring.

The system supports:

- 📹 CCTV streams
- 🎥 Video files
- 📡 RTSP feeds
- 📷 Webcam feeds

to detect fire and smoke incidents in real time with low-latency inference and bounding box localization.

The solution was developed during my internship and deployed in a live monitoring environment for continuous fire surveillance and rapid emergency response.

---

# 🚀 Key Features

- 🔥 Real-time fire detection
- 💨 Smoke detection with localization
- 📹 CCTV, webcam, RTSP & video file support
- ⚡ Optimized YOLO inference pipeline
- 🧠 Custom-trained detection model
- 🚨 Alert-ready architecture
- ☁️ Production deployment support
- 🧩 Scalable modular design
- 🖥️ Live monitoring dashboard integration

---

# 🧠 System Workflow

```text
Input Video Stream
        │
        ▼
Frame Extraction
        │
        ▼
YOLO Fire & Smoke Detection
        │
        ▼
Bounding Box Prediction
        │
        ▼
Alert Trigger & Visualization
        │
        ▼
Real-Time Monitoring Dashboard
```

---

# 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Deep Learning | YOLOv5 / YOLOv8 |
| Programming Language | Python |
| Computer Vision | OpenCV |
| Deep Learning Framework | PyTorch |
| Backend | Flask |
| Annotation Tool | Label Studio |
| Numerical Computing | NumPy |
| Deployment | GPU-enabled Server |

---

# 📂 Project Structure

```text
Fire-Smoke-Detection-System/
│
├── Live_Flask_app/
│   ├── live_detection_dev.py
│   └── live_detection_prod.py
│
├── static_flask_app/
│   ├── static_fire_detection_dev.py
│   └── static_fire_detection_prod.py
│
├── Notebook/
│   └── Fire_Detection (1).ipynb
│
├── results/
│   └── image/
│       ├── download.jpg
│       └── fire-flames-with-smoke-on-black-background-free-photo.jpg
│
├── templates/
│   └── index.html
│
├── README.md
├── Sample_Demo_Video.avi
└── best.pt
```

---

# 📊 Model Capabilities

The YOLO model was custom-trained on manually annotated fire and smoke datasets to improve robustness under diverse environmental conditions:

- Indoor fire scenarios
- Outdoor fire incidents
- Dense smoke environments
- Low-light conditions
- Industrial safety monitoring

---

# 📸 Detection Results

## 🔥 Fire Detection

```text
results/image/download.jpg
```

## 💨 Smoke Detection

```text
results/image/fire-flames-with-smoke-on-black-background-free-photo.jpg
```

---

# ⚙️ Installation & Setup

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/Namanjain122/Fire-Smoke-Detection-System.git
cd Fire-Smoke-Detection-System
```

## 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 3️⃣ Run Live Detection System

```bash
python Live_Flask_app/live_detection_dev.py
```

## 4️⃣ Run Static Detection System

```bash
python static_flask_app/static_fire_detection_dev.py
```

---

# 🎥 Demo

Sample demo video included inside the repository:

```text
Sample_Demo_Video.avi
```

---

# 🌍 Real-World Applications

- 🏭 Industrial safety monitoring
- 🏢 Smart surveillance systems
- 📦 Warehouse fire prevention
- 🌲 Forest fire early detection
- 🏙️ Smart city monitoring
- 🚨 Public safety infrastructure

---

# 🔮 Future Improvements

- 📩 SMS & Email alert integration
- 📲 Telegram / WhatsApp notifications
- 🧠 Edge AI deployment
- 🎥 Multi-camera monitoring support
- ☁️ Cloud dashboard analytics
- 🌡️ Thermal camera integration

---

# 👨‍💻 Author

## Naman Jain

- GitHub: https://github.com/Namanjain122
- LinkedIn: [Link to Post](https://www.linkedin.com/posts/naman-jain-9136732aa_imc25-indiamobilecongress-computervision-activity-7423941683790151681-M3Nj)

---

# ⭐ Acknowledgements

Special thanks to my internship team and the open-source computer vision community for supporting the development of this system.
