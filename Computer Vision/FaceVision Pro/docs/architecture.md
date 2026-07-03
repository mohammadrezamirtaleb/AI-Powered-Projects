# Architecture

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   FaceVision Pro Pipeline                │
│                                                          │
│  Camera / Image / Video                                  │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐    ┌──────────────────────────────┐    │
│  │ FaceDetector │───▶│       Per-Face Analysis       │    │
│  │ (DNN / Haar) │    │                              │    │
│  └─────────────┘    │  ┌──────────────────────┐    │    │
│                      │  │  LandmarkDetector    │    │    │
│                      │  │  (MediaPipe 478-pt)  │    │    │
│                      │  └──────────┬───────────┘    │    │
│                      │             │                 │    │
│                      │  ┌──────────▼───────────┐    │    │
│                      │  │  HeadPoseEstimator   │    │    │
│                      │  │  (solvePnP)          │    │    │
│                      │  └──────────────────────┘    │    │
│                      │                              │    │
│                      │  ┌──────────────────────┐    │    │
│                      │  │  FaceAnalyzer         │    │    │
│                      │  │  (DeepFace:           │    │    │
│                      │  │   emotion/age/gender) │    │    │
│                      │  └──────────────────────┘    │    │
│                      │                              │    │
│                      │  ┌──────────────────────┐    │    │
│                      │  │  FaceRecognizer       │    │    │
│                      │  │  (dlib 128-d embed.)  │    │    │
│                      │  └──────────────────────┘    │    │
│                      └──────────────────────────────┘    │
│         │                                                │
│         ▼                                                │
│  ┌─────────────┐                                         │
│  │ FaceOverlay │  HUD rendering, panels, emotion bars    │
│  └─────────────┘                                         │
│         │                                                │
│         ▼                                                │
│  Output: Display / Video File / Streamlit Dashboard      │
└─────────────────────────────────────────────────────────┘
```

## Module Responsibilities

### `facevision/core/`

| Module | Class | Responsibility |
|---|---|---|
| `detector.py` | `FaceDetector` | Detect face bounding boxes |
| `landmarks.py` | `LandmarkDetector` | Detect 478 facial landmarks, compute EAR |
| `analyzer.py` | `FaceAnalyzer` | Estimate emotion, age, gender via DeepFace |
| `recognizer.py` | `FaceRecognizer` | Encode and match face embeddings |
| `pose.py` | `HeadPoseEstimator` | Compute pitch/yaw/roll from landmarks |

### `facevision/pipeline/`

| Module | Class | Responsibility |
|---|---|---|
| `realtime.py` | `RealtimePipeline` | Webcam loop with keyboard controls |
| `batch.py` | `BatchPipeline` | Image/video file processing |

### `facevision/utils/`

| Module | Class | Responsibility |
|---|---|---|
| `drawing.py` | `FaceOverlay` | OpenCV HUD-style drawing utilities |
| `io.py` | `VideoCapture`, `VideoWriter`, `ImageIO`, `FPSCounter` | I/O helpers |
| `config.py` | `Config` | YAML configuration with dot-key access |

### `facevision/dashboard/`

| Module | Responsibility |
|---|---|
| `app.py` | Streamlit web dashboard for upload-based analysis |
