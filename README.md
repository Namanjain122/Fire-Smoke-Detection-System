🔥 Real-Time Fire and Smoke Detection System using YOLO
📘 Overview

This project is a real-time Fire and Smoke Detection System built using YOLO-based object detection models.

The system is designed to monitor live CCTV feeds, surveillance streams, and video footage to detect fire and smoke incidents in real time and trigger safety alerts for rapid response.

Developed during my internship, this solution is actively deployed in a live environment for continuous monitoring and fire safety operations.

🚀 Features
🔥 Real-time fire detection
💨 Smoke detection with bounding box localization
📹 Supports CCTV feeds, webcams, RTSP streams, and video files
⚡ Optimized YOLO inference for low latency
🧠 Custom-trained model on manually labeled datasets
🚨 Alert-ready architecture for emergency notification systems
☁️ Production deployment support
🧩 Modular architecture for scalability and integration
🧠 System Architecture
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
🧑‍💻 Tech Stack
Category	Technologies
Deep Learning	YOLOv5 / YOLOv8
Programming Language	Python
Computer Vision	OpenCV
Deep Learning Framework	PyTorch
Annotation Tool	Label Studio
Backend/API	Flask
Numerical Computing	NumPy
Deployment	GPU-enabled Server
📂 Project Structure
fire-smoke-detection-system/
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
📊 Model Capabilities

The YOLO model was custom-trained on fire and smoke datasets with manually annotated images to improve detection robustness under different environmental conditions such as:

Indoor fire
Outdoor fire
Dense smoke
Low-light environments
Industrial monitoring scenarios
📸 Sample Results
🔥 Fire Detection

results/image/download.jpg

💨 Smoke Detection

results/image/fire-flames-with-smoke-on-black-background-free-photo.jpg

⚙️ Installation
Clone Repository
git clone https://github.com/Namanjain122/Fire-Smoke-Detection-System.git
cd Fire-Smoke-Detection-System
Install Dependencies
pip install -r requirements.txt
Run Live Detection System
python Live_Flask_app/live_detection_dev.py
Run Static Detection System
python static_flask_app/static_fire_detection_dev.py
🎥 Demo

Sample demo video included:

Sample_Demo_Video.avi
🌍 Real-World Applications
Smart surveillance systems
Industrial safety monitoring
Warehouse fire prevention
Forest fire early detection
Smart city monitoring
Public safety systems
🔮 Future Improvements
SMS / Email alert integration
Telegram / WhatsApp notifications
Edge AI deployment
Multi-camera monitoring support
Cloud dashboard analytics
Thermal camera integration

👨‍💻 Author
Naman Jain
GitHub: Namanjain122 GitHub
- LinkedIn: [Link to post](https://www.linkedin.com/posts/naman-jain-9136732aa_imc25-indiamobilecongress-computervision-activity-7423941683790151681-M3Nj?utm_source=share&utm_medium=member_desktop&rcm=ACoAAEp-OF8BoZi6dSyYN5Xrf1kujyocZc_kzTM)

---

# ⭐ Acknowledgements

Special thanks to my internship team and the open-source computer vision community for supporting the development of this system.
