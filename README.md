# RoadVision AI – Road Object Detection

## Project Overview
RoadVision AI is an AI-powered object detection application developed using Python, YOLOv8, OpenCV, and Streamlit.

The application detects supported objects in uploaded road images and displays bounding boxes, object labels, and confidence scores.

This project demonstrates the practical implementation of computer vision and deep learning in an interactive web application.

## 🎯 Objectives
- Detect objects in road images using AI.
- Identify supported objects such as cars, buses, trucks, and people.
- Display detected objects with bounding boxes.
- Provide an interactive user interface for image-based detection.
- Explore computer vision applications for road environments.

## ✨ Features
- 📷 Image upload for object detection
- 🤖 YOLOv8 pretrained model integration
- 🔍 Object detection and classification
- 📦 Bounding box visualization
- 📊 Detection confidence scores
- 🖥️ Interactive Streamlit interface

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| YOLOv8 | Object detection |
| OpenCV | Image processing |
| Streamlit | Web application interface |
| NumPy | Numerical operations |
| Pillow | Image handling |

## 🧠 How It Works

1. User uploads a road image.
2. The image is processed by the YOLOv8 model.
3. The model detects supported objects.
4. Bounding boxes and class labels are generated.
5. Detection results and confidence scores are displayed.

## 📂 Project Structure

```text
RoadVision-AI/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── START_HERE.txt
└── screenshots/
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/RoadVision-AI.git
```

### 2. Navigate to the project folder

```bash
cd RoadVision-AI
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 5. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 6. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 📸 Application Screenshots

Add your project screenshots here.

Example:

- Homepage
- Uploaded road image
- Object detection result with bounding boxes
- Detection results and confidence scores

## 📊 Model Information

This project uses a pretrained YOLOv8 model for object detection.

The model detects objects belonging to the classes it was trained to recognize. Detection performance may vary depending on image quality, object size, lighting, and visibility.

##  Future Improvements

- Real-time webcam detection
- Video-based object detection
- Traffic object analysis
- Model performance evaluation
- Improved detection accuracy
- Deployment as a publicly accessible web application

##  Learning Outcomes

Through this project, I gained practical experience in:

- Computer vision
- Object detection using YOLO
- Python application development
- Image processing using OpenCV
- Integrating AI models into web applications
- Building interactive applications using Streaml

**Sakshi Thombre**

Aspiring AI/ML Engineer | MCA Pursuing

GitHub: https://github.com/sakshichandrakantthombre-debug

---

 If you find this project interesting, feel free to explore it and share your feedback!
