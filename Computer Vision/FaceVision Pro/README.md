# 👤 FaceVision Pro

<div align="center">

[![CI](https://github.com/yourusername/FaceVisionPro/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/FaceVisionPro/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)](https://opencv.org)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-red)](https://mediapipe.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**A professional, modular Computer Vision toolkit for real-time face analysis.**

[Features](#-features) · [Installation](#-installation) · [Quick Start](#-quick-start) · [Dashboard](#-streamlit-dashboard) · [Architecture](#-architecture) · [Contributing](#-contributing)

</div>

---

## ✨ Features

| Feature | Technology | Description |
|---|---|---|
| 🎯 **Face Detection** | OpenCV DNN (SSD ResNet) | Robust detection across angles and lighting |
| 🗺️ **Facial Landmarks** | MediaPipe Face Mesh | 478-point landmark detection |
| 👁️ **Blink Detection** | Eye Aspect Ratio (EAR) | Real-time blink counting |
| 😀 **Emotion Analysis** | DeepFace | 7-class emotion classification |
| 🎂 **Age Estimation** | DeepFace | Per-frame age prediction |
| 🧑 **Gender Detection** | DeepFace | Gender classification with confidence |
| 🧑‍🤝‍🧑 **Face Recognition** | dlib (128-d embeddings) | Identify enrolled people in real time |
| 📐 **Head Pose** | OpenCV solvePnP | Pitch, Yaw, Roll in degrees |
| 🎬 **Video Recording** | OpenCV VideoWriter | One-key recording with timestamp |
| 🖥️ **Streamlit Dashboard** | Streamlit | Upload & analyze images/videos via browser |
| ⚙️ **YAML Config** | PyYAML | Fully configurable pipeline |
| 🤖 **GitHub Actions CI** | GitHub Actions | Lint + test on every push |

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- A webcam (for real-time mode)
- pip

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/FaceVisionPro.git
cd FaceVisionPro
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Download pre-trained model weights

```bash
python scripts/download_models.py
```

> This downloads the OpenCV SSD ResNet face detector model (~10 MB) into the `models/` directory.
> If you skip this step, the pipeline automatically falls back to the built-in Haar Cascade detector.

---

## ⚡ Quick Start

### Real-Time Webcam Pipeline

```bash
python scripts/run_webcam.py
```

**Keyboard controls:**

| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `SPACE` | Pause / Resume |
| `L` | Toggle landmark overlay |
| `P` | Toggle head pose axes |
| `E` | Toggle emotion bars |
| `S` | Save snapshot |
| `R` | Start / Stop recording |

### Analyze a Single Image

```bash
python scripts/run_image.py --image photo.jpg --show
```

### Enroll Your Face (for Recognition)

```bash
# Enroll from webcam (captures 5 frames)
python scripts/enroll_face.py --name "Your Name" --camera 0

# Enroll from image files
python scripts/enroll_face.py --name "Your Name" --images photo1.jpg photo2.jpg

# List all enrolled people
python scripts/enroll_face.py --list

# Remove a person
python scripts/enroll_face.py --remove "Your Name"
```

### Python API

```python
from facevision import FaceDetector, LandmarkDetector, FaceAnalyzer
import cv2

# Initialize modules
detector = FaceDetector(method="dnn")
landmarks = LandmarkDetector()
analyzer = FaceAnalyzer(actions=["emotion", "age", "gender"])

# Load an image
image = cv2.imread("photo.jpg")

# Detect faces
faces = detector.detect(image)
print(f"Found {len(faces)} face(s)")

# Analyze the first face
if faces:
    crop = faces[0].crop(image)
    result = analyzer.analyze(crop, face_id=0)
    print(f"Emotion: {result.dominant_emotion}")
    print(f"Age: ~{result.age}")
    print(f"Gender: {result.gender}")

# Detect landmarks
lm_results = landmarks.detect(image)
if lm_results:
    nose_tip = lm_results[0].get_point(1)  # MediaPipe landmark #1
    print(f"Nose tip: {nose_tip}")
```

---

## 🖥️ Streamlit Dashboard

Launch the interactive web dashboard:

```bash
streamlit run facevision/dashboard/app.py
```

Then open http://localhost:8501 in your browser.

**Dashboard features:**
- 📸 Upload images for instant analysis
- 🎥 Upload video files for batch processing
- Interactive sliders and toggles for all pipeline settings
- Per-face breakdown panels with emotion probability bars
- Head pose angle display

---

## ⚙️ Configuration

Edit `configs/default.yaml` to customize the pipeline:

```yaml
detector:
  method: dnn              # "dnn" or "haar"
  confidence_threshold: 0.60

analyzer:
  enabled: true
  actions: [emotion, age, gender]
  cooldown_seconds: 1.5    # Re-analyze each face every N seconds

recognizer:
  enabled: true
  tolerance: 0.50          # Lower = stricter matching

pipeline:
  camera_index: 0
  width: 1280
  height: 720
  flip_horizontal: true
```

Or pass overrides via command line:

```bash
python scripts/run_webcam.py --method haar --no-analysis --camera 1
```

---

## 🏗️ Architecture

```
FaceVision Pro/
├── .github/
├── assets/
├── configs/
├── data/
├── docs/
├── facevision/
├── models/
├── scripts/
├── tests/
├── pyproject.toml
├── setup.py
├── setup.cfg
...
```

---

## 🧪 Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=facevision --cov-report=html
open htmlcov/index.html
```

---

## 🤝 Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Commit: `git commit -m "feat: add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

---

## 👨‍💻 Author
Mohammad Reza Mirtaleb

MSC at Petroleum University of Technology, Abadan Faculty

AI Engineer | Machine Learning & Deep Learning Engineer | Data Scientist | NLP Expert (LLMs and VLMs) | RAG and Multi-Agent Systems Developer

Building intelligent solutions for real-world challenges.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">
Made with ❤️ using OpenCV · MediaPipe · DeepFace · dlib
</div>
