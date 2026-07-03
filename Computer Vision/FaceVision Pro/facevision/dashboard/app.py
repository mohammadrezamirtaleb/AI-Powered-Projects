"""
Streamlit Dashboard for FaceVision Pro
========================================
Interactive web dashboard for uploading and analyzing images/videos.

Run:
    streamlit run facevision/dashboard/app.py
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FaceVision Pro",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { background: #0d0f14; }
    .stApp { background: linear-gradient(135deg, #0d0f14 0%, #141824 100%); }

    /* Title */
    .title-block {
        background: linear-gradient(90deg, #00e5ff, #7c4dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -1px;
        margin-bottom: 0;
    }
    .subtitle { color: #8892a4; font-size: 1rem; margin-top: -0.5rem; }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(0,229,255,0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        backdrop-filter: blur(8px);
    }
    .metric-label { color: #8892a4; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #00e5ff; font-size: 1.4rem; font-weight: 600; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0a0c10; border-right: 1px solid #1e2332; }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00e5ff22, #7c4dff22);
        border: 1px solid #00e5ff55;
        color: #00e5ff;
        border-radius: 8px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #00e5ff44, #7c4dff44);
        border-color: #00e5ff;
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(0,229,255,0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Cached module loading ───────────────────────────────────────────────────────
@st.cache_resource
def load_modules():
    from facevision.core.analyzer import FaceAnalyzer
    from facevision.core.detector import FaceDetector
    from facevision.core.landmarks import LandmarkDetector
    from facevision.core.pose import HeadPoseEstimator
    from facevision.core.recognizer import FaceRecognizer
    from facevision.utils.drawing import FaceOverlay

    detector = FaceDetector(method="dnn", confidence_threshold=0.5)
    landmark_detector = LandmarkDetector(max_faces=10)
    analyzer = FaceAnalyzer(actions=["emotion", "age", "gender"], cooldown_seconds=0)
    recognizer = FaceRecognizer()
    recognizer.load_database()
    pose_estimator = HeadPoseEstimator(smoothing=1)
    overlay = FaceOverlay(alpha=0.80)
    return detector, landmark_detector, analyzer, recognizer, pose_estimator, overlay


def analyze_image(image_bgr: np.ndarray, modules, options: dict) -> tuple:
    """Run the full pipeline on one image."""
    detector, lm_detector, analyzer, recognizer, pose_est, overlay = modules
    annotated = image_bgr.copy()
    faces = detector.detect(annotated)
    all_landmarks = lm_detector.detect(annotated) if options["show_landmarks"] else []
    results = []
    for i, face in enumerate(faces):
        crop = face.crop(annotated)
        analysis = analyzer.analyze(crop, i) if options["show_analysis"] and crop.size > 0 else None
        recognition = recognizer.recognize(crop, i) if options["show_recognition"] and crop.size > 0 else None
        lm_result = all_landmarks[i] if i < len(all_landmarks) else None
        pose = pose_est.estimate(lm_result, annotated.shape) if options["show_pose"] and lm_result else None
        overlay.draw_face_box(annotated, face, label=recognition.display_name if recognition else None)
        if lm_result and options["show_landmarks"]:
            overlay.draw_landmarks(annotated, lm_result)
        overlay.draw_analysis_panel(annotated, face, analysis, recognition, pose, lm_result)
        results.append({"face": face, "analysis": analysis, "recognition": recognition, "pose": pose})
    overlay.draw_watermark(annotated)
    return annotated, results, faces


# ── Layout ─────────────────────────────────────────────────────────────────────

def main():
    # Header
    st.markdown('<p class="title-block">FaceVision Pro</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Professional Real-Time Face Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        show_landmarks = st.toggle("Show Landmarks", value=True)
        show_analysis = st.toggle("Show Analysis", value=True)
        show_recognition = st.toggle("Show Recognition", value=True)
        show_pose = st.toggle("Show Head Pose", value=True)

        st.markdown("---")
        st.markdown("### 🎯 Detector")
        method = st.selectbox("Method", ["dnn", "haar"], index=0)
        confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.6, 0.05)

        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown(
            "**FaceVision Pro v1.0.0**\n\n"
            "A professional computer vision toolkit for face analysis.\n\n"
            "🔗 [GitHub](https://github.com)"
        )

    options = {
        "show_landmarks": show_landmarks,
        "show_analysis": show_analysis,
        "show_recognition": show_recognition,
        "show_pose": show_pose,
    }

    # Load modules
    with st.spinner("Loading AI models …"):
        modules = load_modules()

    tab1, tab2 = st.tabs(["📸 Image Analysis", "🎥 Video Analysis"])

    with tab1:
        uploaded = st.file_uploader(
            "Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), np.uint8)
            img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img_bgr is None:
                st.error("Could not decode the uploaded image.")
            else:
                with st.spinner("Analyzing …"):
                    t0 = time.perf_counter()
                    annotated, results, faces = analyze_image(img_bgr, modules, options)
                    elapsed = time.perf_counter() - t0

                # Results
                col_img, col_info = st.columns([3, 2])
                with col_img:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, caption="Annotated Result", use_container_width=True)

                with col_info:
                    st.markdown(f"**⏱ Analysis time:** `{elapsed*1000:.0f} ms`")
                    st.markdown(f"**👤 Faces detected:** `{len(faces)}`")
                    st.markdown(f"**🔍 Detector:** `{method.upper()}`")
                    st.markdown("---")
                    for i, res in enumerate(results):
                        with st.expander(f"Face #{i+1}", expanded=True):
                            st.markdown(
                                f'<div class="metric-card">'
                                f'<div class="metric-label">Name</div>'
                                f'<div class="metric-value">{res["recognition"].display_name if res["recognition"] else "—"}</div>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
                            if res["analysis"] and res["analysis"].success:
                                a = res["analysis"]
                                cols = st.columns(3)
                                cols[0].metric("Emotion", f"{a.emotion_emoji} {a.dominant_emotion.title()}")
                                cols[1].metric("Age", f"~{a.age}")
                                cols[2].metric("Gender", a.gender)
                                if a.emotion_scores:
                                    st.markdown("**Emotion probabilities:**")
                                    for emo, score in sorted(a.emotion_scores.items(), key=lambda x: x[1], reverse=True)[:4]:
                                        val = score / 100.0 if score > 1 else score
                                        st.progress(val, text=f"{emo.title()}: {val:.0%}")
                            if res["pose"] and res["pose"].success:
                                p = res["pose"]
                                pc1, pc2, pc3 = st.columns(3)
                                pc1.metric("Yaw", f"{p.yaw:+.1f}°")
                                pc2.metric("Pitch", f"{p.pitch:+.1f}°")
                                pc3.metric("Roll", f"{p.roll:+.1f}°")
                                st.info(f"🧭 {p.direction}")

    with tab2:
        st.markdown("#### 🎥 Video File Analysis")
        vid_uploaded = st.file_uploader(
            "Upload a video", type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed", key="video_upload",
        )
        if vid_uploaded:
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
                tmp.write(vid_uploaded.read())
                tmp_path = Path(tmp.name)

            st.video(str(tmp_path))
            if st.button("▶ Analyze Video"):
                from facevision.pipeline.batch import BatchPipeline
                pipeline = BatchPipeline(output_dir=Path("outputs/dashboard"))
                with st.spinner("Processing video …"):
                    out = pipeline.process_video(tmp_path, save=True, max_frames=300)
                if out and out.exists():
                    st.success(f"✅ Video processed! Saved to `{out}`")
                    st.video(str(out))
                else:
                    st.error("Video processing failed.")


if __name__ == "__main__":
    main()
