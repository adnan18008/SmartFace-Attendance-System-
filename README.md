# Face Recognition Attendance System 👤📷

Face Recognition Attendance System is an AI-based application that automatically detects and recognizes faces using a webcam and records attendance in a database. The system uses computer vision and deep learning–based face embeddings to identify registered users and provides a simple web dashboard to view attendance records.

## ✨ Features

* **Real-time Face Recognition:** Detects and identifies faces using Dlib and OpenCV.
* **Automatic Attendance Logging:** Records attendance directly into a SQLite database.
* **Face Enrollment GUI:** Register new users by capturing face images through the camera.
* **Attendance Dashboard:** View attendance records by date using a Flask web interface.

## ⬇️ How to Download

You can download the source code from GitHub to run it locally.

1. **Clone the repository**

```bash
git clone https://github.com/your-username/face-recognition-attendance-system.git
```

2. **Navigate to the project folder**

```bash
cd face-recognition-attendance-system
```

## ⚙️ Setup Environment

1. **Create a virtual environment**

```bash
python -m venv .venv
```

2. **Activate the environment**

**Windows**

```bash
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

## 📦 Required Model Files

Download the following Dlib model files and place them in `data/data_dlib/`.

* **shape_predictor_68_face_landmarks.dat**
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

* **dlib_face_recognition_resnet_model_v1.dat**
  http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Extract the `.bz2` files before placing them in the folder.

## 🚀 How to Run

### 1️⃣ Enroll Faces

```bash
python face_enrollment.py
```

### 2️⃣ Generate Face Embeddings

```bash
python embedding_extractor.py
```

### 3️⃣ Start Real-Time Recognition

```bash
python realtime_recognizer.py
```

### 4️⃣ Open the Attendance Dashboard

```bash
python app.py
```

Then open:

```
http://127.0.0.1:5000
```

## 🧠 Technologies Used

* Python
* OpenCV
* Dlib
* NumPy & Pandas
* Flask
* SQLite

## 📂 Project Structure

```
Face-Recognition-Attendance-System
│
├── config
│   └── paths.py
├── data
│   ├── data_dlib
│   └── data_faces_from_camera
├── templates
│   └── index.html
├── app.py
├── face_enrollment.py
├── embedding_extractor.py
├── realtime_recognizer.py
├── requirements.txt
├── README.md
└── LICENSE
```

## 📜 License

This project is licensed under the **MIT License**.
