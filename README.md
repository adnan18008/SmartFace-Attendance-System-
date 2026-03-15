<<<<<<< HEAD
# Face_Recognition_Attendance_System
AI-powered face recognition attendance system using Python, OpenCV, and Dlib that automatically detects, recognizes, and records attendance in a database with a simple web interface.
=======
# Face Recognition Attendance System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Dlib](https://img.shields.io/badge/Dlib-face%20recognition-orange.svg)](http://dlib.net/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-style **face recognition attendance system** that identifies people in real time from a webcam, logs attendance to SQLite, and provides a web dashboard to view records by date. Built with **Dlib** (68-point landmarks + ResNet-based 128D embeddings), **OpenCV**, and **Flask**.

---

## Highlights

- **Deep learning–based recognition**: 128D face embeddings via Dlib’s ResNet model; Euclidean distance matching with a configurable threshold (default 0.4).
- **Multi-face tracking**: Centroid-based tracking across frames to keep identities stable and avoid duplicate attendance entries.
- **End-to-end pipeline**: Enrollment (GUI) → embedding extraction (CSV) → real-time recognition + attendance logging → web dashboard.
- **Single entry per person per day**: SQLite schema enforces one attendance record per identity per date.
- **Clean project layout**: Centralized paths in `config`, separate scripts for enrollment, embedding build, recognition, and web app.

---

## Architecture

```
┌─────────────────────┐     ┌──────────────────────┐     ┌─────────────────────────┐
│  face_enrollment.py  │────▶│ embedding_extractor  │────▶│  realtime_recognizer.py  │
│  (camera + GUI)      │     │ (128D → features CSV)│     │  (camera + DB logging)  │
└─────────────────────┘     └──────────────────────┘     └────────────┬────────────┘
         │                              │                             │
         ▼                              ▼                             ▼
  data/data_faces_from_camera/    data/features_all.csv         attendance.db
  (per-person face images)        (known embeddings)            (attendance records)
                                                                       │
                                                                       ▼
                                                              ┌─────────────────────┐
                                                              │  app.py (Flask)     │
                                                              │  View by date       │
                                                              └─────────────────────┘
```

---

## Tech Stack

| Component        | Technology |
|-----------------|------------|
| Face detection  | Dlib (HOG + 68-point landmarks) |
| Face embeddings  | Dlib ResNet (128D descriptor) |
| Matching        | L2 (Euclidean) distance, threshold 0.4 |
| Tracking        | Centroid-based frame-to-frame association |
| Backend         | Python 3.8+, OpenCV, NumPy, Pandas |
| Storage         | SQLite (attendance), CSV (embeddings) |
| Web             | Flask, Jinja2, Bootstrap 5 |

---

## Project Structure

```
.
├── config/
│   ├── __init__.py
│   └── paths.py          # Central path config (data, models, DB)
├── data/
│   ├── data_dlib/        # Dlib model files (see Setup)
│   ├── data_faces_from_camera/  # Enrolled face images (per person)
│   └── features_all.csv  # 128D embeddings for known faces
├── templates/
│   └── index.html        # Attendance dashboard UI
├── realtime_recognizer.py   # Live recognition + attendance logging
├── face_enrollment.py       # GUI to capture faces per identity
├── embedding_extractor.py   # Build features_all.csv from enrolled images
├── app.py                   # Flask app to view attendance by date
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and environment

```bash
git clone <your-repo-url>
cd Face_Recogniton_Attendance_system
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Dlib model files

Place the following files under `data/data_dlib/`:

- **Shape predictor (68 face landmarks)**  
  Download: [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)  
  Decompress and save as `data/data_dlib/shape_predictor_68_face_landmarks.dat`

- **Face recognition model (ResNet)**  
  Download: [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)  
  Decompress and save as `data/data_dlib/dlib_face_recognition_resnet_model_v1.dat`

### 3. Run from project root

All commands assume your current directory is the project root (where `config/` and `app.py` live).

---

## Usage

### Step 1: Enroll faces

```bash
python face_enrollment.py
```

- Use the GUI: enter a name, click **Input**, then **Save current face** when a single face is clearly visible.
- Repeat for each person (and optionally multiple shots per person).
- Face images are stored under `data/data_faces_from_camera/` (e.g. `person_1_Alice/`, `person_2_Bob/`).

### Step 2: Build embedding database

```bash
python embedding_extractor.py
```

- Reads all person folders in `data/data_faces_from_camera/`, computes 128D embeddings, and writes `data/features_all.csv`.
- This CSV is the “known faces” database for the recognizer.

### Step 3: Run real-time recognition and attendance

```bash
python realtime_recognizer.py
```

- Opens the default camera, detects faces, matches them to the CSV embeddings (L2 &lt; 0.4), and logs attendance to `attendance.db` (one record per person per day).
- Press **Q** to quit.

### Step 4: View attendance in the dashboard

```bash
python app.py
```

- Open the URL shown (e.g. `http://127.0.0.1:5000`), pick a date, and view the attendance table for that day.

---

## Configuration

- **Paths**: Edit `config/paths.py` to change data dir, model paths, or DB path.
- **Recognition threshold**: In `realtime_recognizer.py`, the match threshold is `0.4` (smaller = stricter). Adjust the condition `min_dist < 0.4` if needed.
- **Camera**: Default camera ID is `0`. You can pass another ID into `RealtimeFaceRecognizer().run(camera_id=...)` if needed.

---

## Requirements

- Python 3.8+
- Webcam
- See `requirements.txt` for package versions (e.g. `dlib`, `opencv-python`, `numpy`, `pandas`, `flask`).

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Author

Built as a portfolio project demonstrating **computer vision**, **face recognition**, and **full-stack integration** (desktop GUI, CLI pipeline, web dashboard, SQLite).
>>>>>>> d3bbe24 (Initial project commit)
