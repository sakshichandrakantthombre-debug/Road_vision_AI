import os
import sqlite3
import hashlib
import hmac
import secrets
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "users.db"
MODEL_NAME = os.getenv("YOLO_MODEL", "yolov8n.pt")

st.set_page_config(
    page_title="RoadVision AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Authentication helpers ----------
def db_connect():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with db_connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()

def hash_password(password, salt=None):
    salt_bytes = bytes.fromhex(salt) if salt else secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 250_000)
    return salt_bytes.hex(), digest.hex()

def create_user(username, email, password):
    username = username.strip()
    email = email.strip().lower()
    if len(username) < 3:
        return False, "Username must contain at least 3 characters."
    if "@" not in email or "." not in email.split("@")[-1]:
        return False, "Please enter a valid email address."
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    salt, digest = hash_password(password)
    try:
        with db_connect() as conn:
            conn.execute(
                "INSERT INTO users (username, email, salt, password_hash) VALUES (?, ?, ?, ?)",
                (username, email, salt, digest),
            )
            conn.commit()
        return True, "Account created. Please sign in."
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."

def verify_user(login, password):
    with db_connect() as conn:
        row = conn.execute(
            "SELECT username, email, salt, password_hash FROM users WHERE username = ? COLLATE NOCASE OR email = ? COLLATE NOCASE",
            (login.strip(), login.strip()),
        ).fetchone()
    if not row:
        return None
    username, email, salt, stored_hash = row
    _, candidate = hash_password(password, salt)
    if hmac.compare_digest(candidate, stored_hash):
        return {"username": username, "email": email}
    return None

@st.cache_resource
def load_model():
    # The first run downloads the small pretrained YOLO model automatically.
    return YOLO(MODEL_NAME)

def run_detection(image_rgb, confidence):
    model = load_model()
    results = model.predict(source=image_rgb, conf=confidence, verbose=False)
    result = results[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    rows = []
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            xyxy = [round(float(v), 1) for v in box.xyxy[0].tolist()]
            rows.append({
                "Object": names.get(cls_id, str(cls_id)),
                "Confidence": round(conf, 3),
                "Box (x1, y1, x2, y2)": str(xyxy),
            })
    return annotated_rgb, rows

def logout():
    for key in ("authenticated", "username", "email"):
        st.session_state.pop(key, None)
    st.rerun()

init_db()

# ---------- Visual style ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(145deg, #08111f 0%, #0d1b2e 48%, #10283a 100%); color: #f4f7fb; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    [data-testid="stSidebar"] { background: #0a1423; border-right: 1px solid #20364a; }
    h1, h2, h3 { color: #f5f8ff !important; }
    p, label, .stMarkdown { color: #d5e1ef; }
    .hero { padding: 1.5rem 1.7rem; border: 1px solid #28465b; border-radius: 22px;
            background: linear-gradient(120deg, rgba(20,48,68,.95), rgba(12,27,47,.95)); margin-bottom: 1.2rem; }
    .hero-kicker { color: #50e3c2; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; font-size: .78rem; }
    .muted { color: #a8bbcf; }
    div[data-testid="stMetric"] { background: #10263a; border: 1px solid #28465b; padding: 14px; border-radius: 14px; }
    .stButton > button { border-radius: 12px; border: 1px solid #3bdbc0; background: #21b99f; color: #071521; font-weight: 700; }
    .stButton > button:hover { background: #4be0c6; border-color: #4be0c6; color: #071521; }
    div[data-testid="stForm"] { border: 1px solid #28465b; border-radius: 16px; padding: 1rem; background: rgba(13,31,49,.7); }
    .small-note { font-size: .84rem; color: #9fb4c9; }
</style>
""", unsafe_allow_html=True)

# ---------- Auth screen ----------
if not st.session_state.get("authenticated"):
    st.markdown("""
    <div class="hero">
      <div class="hero-kicker">Computer Vision • Road Safety</div>
      <h1 style="font-size:2.7rem; margin:.35rem 0;">🚦 RoadVision AI</h1>
      <p style="font-size:1.1rem;">See the road differently. Detect vehicles, people, and everyday objects with AI.</p>
      <p class="muted">A portfolio-ready computer vision demo built with YOLO, OpenCV, and Streamlit.</p>
    </div>
    """, unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.subheader("What you can do")
        st.markdown("""
        - **Image detection:** upload a road photo and view bounding boxes.
        - **Video detection:** analyze a short road clip frame by frame.
        - **Webcam detection:** run live detection from your local webcam.
        - **Detection report:** review predicted classes and confidence scores.
        """)
        st.info("The starter model recognizes its pretrained COCO classes. It cannot identify every possible object or guarantee road safety.")
        st.markdown('<p class="small-note">Privacy note: uploaded media is processed by this app; this starter does not intentionally save uploaded images or videos.</p>', unsafe_allow_html=True)
    with right:
        tab_login, tab_signup = st.tabs(["Sign in", "Create account"])
        with tab_login:
            with st.form("login_form"):
                login = st.text_input("Username or email", key="login_identifier")
                password = st.text_input("Password", type="password", key="login_password")
                submitted = st.form_submit_button("Sign in", use_container_width=True)
            if submitted:
                user = verify_user(login, password)
                if user:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = user["username"]
                    st.session_state["email"] = user["email"]
                    st.rerun()
                else:
                    st.error("Sign-in failed. Check your username/email and password.")
        with tab_signup:
            with st.form("signup_form"):
                new_username = st.text_input("Choose a username", key="signup_username")
                new_email = st.text_input("Email address", key="signup_email")
                new_password = st.text_input("Create password (8+ characters)", type="password", key="signup_password")
                confirm_password = st.text_input("Confirm password", type="password", key="signup_confirm")
                create = st.form_submit_button("Create account", use_container_width=True)
            if create:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    ok, message = create_user(new_username, new_email, new_password)
                    if ok:
                        st.success(message)
                    else:
                        st.error(message)
    st.stop()

# ---------- Main app ----------
with st.sidebar:
    st.markdown("## 🚦 RoadVision AI")
    st.caption("AI-powered road scene explorer")
    st.markdown("---")
    st.write(f"Welcome, **{st.session_state.get('username', 'User')}** 👋")
    page = st.radio("Navigate", ["Dashboard", "Image detection", "Video detection", "Webcam detection", "About"])
    st.markdown("---")
    st.caption("YOLO • OpenCV • Streamlit")
    if st.button("Sign out", use_container_width=True):
        logout()

st.markdown(f"""
<div class="hero">
  <div class="hero-kicker">Welcome back</div>
  <h1 style="font-size:2.35rem; margin:.3rem 0;">Hello, {st.session_state.get('username', 'there')} 👋</h1>
  <p>Ready to explore a road scene? Choose a detection mode and let the model analyze it.</p>
</div>
""", unsafe_allow_html=True)

confidence = st.sidebar.slider("Detection confidence", min_value=0.10, max_value=0.90, value=0.35, step=0.05,
                               help="Higher confidence reduces weaker detections.")

if page == "Dashboard":
    st.subheader("Your detection workspace")
    c1, c2, c3 = st.columns(3)
    c1.metric("Detection modes", "3", "Image • Video • Webcam")
    c2.metric("Model", MODEL_NAME, "Pretrained YOLO")
    c3.metric("Access", "Signed in", "Local account")
    st.markdown("### Start here")
    a, b, c = st.columns(3)
    with a:
        st.markdown("#### 🖼️ Image")
        st.write("Analyze a JPG, JPEG, or PNG road image.")
        if st.button("Open image detection", key="go_image", use_container_width=True):
            st.session_state["nav_target"] = "Image detection"
            st.rerun()
    with b:
        st.markdown("#### 🎞️ Video")
        st.write("Process a short MP4 or MOV clip.")
        if st.button("Open video detection", key="go_video", use_container_width=True):
            st.session_state["nav_target"] = "Video detection"
            st.rerun()
    with c:
        st.markdown("#### 📷 Webcam")
        st.write("Run live detection using your computer's camera.")
        if st.button("Open webcam detection", key="go_webcam", use_container_width=True):
            st.session_state["nav_target"] = "Webcam detection"
            st.rerun()
    if st.session_state.get("nav_target"):
        st.info("Use the Navigate menu in the sidebar to select " + st.session_state["nav_target"] + ".")
    st.markdown("---")
    st.markdown("**Portfolio note:** add your own test images, record honest metrics, and explain the model's limitations in your README.")

elif page == "Image detection":
    st.subheader("🖼️ Image detection")
    uploaded = st.file_uploader("Upload a road image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner("Detecting objects…"):
            try:
                annotated, rows = run_detection(np.array(image), confidence)
                original_col, result_col = st.columns(2)
                with original_col:
                    st.markdown("**Original image**")
                    st.image(image, use_container_width=True)
                with result_col:
                    st.markdown("**AI detection result**")
                    st.image(annotated, use_container_width=True)
                st.success(f"Detection complete — {len(rows)} object(s) found.")
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    csv = "Object,Confidence,Box (x1, y1, x2, y2)\n" + "\n".join(
                        f'"{r["Object"]}",{r["Confidence"]},"{r["Box (x1, y1, x2, y2)"]}"' for r in rows
                    )
                    st.download_button("Download detection report (CSV)", csv, "roadvision_detections.csv", "text/csv")
                else:
                    st.warning("No objects met the selected confidence threshold. Try lowering it.")
            except Exception as exc:
                st.error("Detection failed. Check your installation and internet connection for the first model download.")
                st.exception(exc)

elif page == "Video detection":
    st.subheader("🎞️ Video detection")
    st.caption("For smoother performance on an 8 GB RAM laptop, start with a short, low-resolution clip.")
    video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    max_frames = st.slider("Maximum frames to analyze", 10, 120, 40, 10)
    if video:
        temp_path = APP_DIR / "_uploaded_video_temp.mp4"
        temp_path.write_bytes(video.getvalue())
        cap = cv2.VideoCapture(str(temp_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
        step = max(1, total // max_frames) if total else 1
        frames_rgb = []
        progress = st.progress(0)
        status = st.empty()
        try:
            index, checked = 0, 0
            while cap.isOpened() and checked < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    small = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1]))) if frame.shape[1] > 640 else frame
                    frame_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                    annotated, rows = run_detection(frame_rgb, confidence)
                    frames_rgb.append(annotated)
                    checked += 1
                    progress.progress(min(checked / max_frames, 1.0))
                    status.write(f"Processed frame {checked} of up to {max_frames}")
                index += 1
            if frames_rgb:
                st.success(f"Processed {len(frames_rgb)} frames. Showing a sample of the annotated output.")
                st.image(frames_rgb[:min(6, len(frames_rgb))], caption=[f"Processed frame {i+1}" for i in range(min(6, len(frames_rgb)))])
                st.caption(f"Original clip FPS reported by file: {fps:.1f}")
            else:
                st.warning("Could not read frames from this video. Try a different MP4 file.")
        except Exception as exc:
            st.error("Video processing failed.")
            st.exception(exc)
        finally:
            cap.release()
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

elif page == "Webcam detection":
    st.subheader("📷 Live webcam detection")
    st.warning("Webcam access is intended for running this app locally. Your browser may ask you to allow camera access.")
    st.markdown("If you deploy this app publicly, webcam support depends on browser permissions, HTTPS, and the hosting environment.")
    try:
        from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
        import av

        model = load_model()
        class VideoProcessor:
            def __init__(self):
                self.last_error = None
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                try:
                    result = model.predict(source=img, conf=confidence, verbose=False)[0]
                    annotated = result.plot()
                    return av.VideoFrame.from_ndarray(annotated, format="bgr24")
                except Exception:
                    return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="roadvision-live-camera",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
            async_processing=True,
        )
        st.caption("Tip: close other apps that use the camera. On a slower laptop, use good lighting and a small camera resolution.")
    except ImportError:
        st.error("Webcam dependency is missing. Run: pip install streamlit-webrtc av")
    except Exception as exc:
        st.error("Could not start webcam detection. Check camera permission and dependency installation.")
        st.exception(exc)

elif page == "About":
    st.subheader("About RoadVision AI")
    st.write("RoadVision AI is a learning and portfolio project that demonstrates object detection with a pretrained YOLO model.")
    st.markdown("""
    **Included**
    - Local sign-up and sign-in with salted PBKDF2 password hashes.
    - SQLite user database.
    - Image, short-video, and webcam detection modes.
    - Adjustable confidence threshold.
    - Downloadable image-detection report.
    - Dark, responsive Streamlit interface.

    **Important limitations**
    - The pretrained COCO model recognizes a fixed list of classes, not every possible road object.
    - This is not a certified road-safety or autonomous-driving system.
    - This starter authentication is suitable for a local portfolio demo, not a public production service without security review, HTTPS, rate limiting, account recovery, and managed secrets.
    - Do not commit `users.db`, secrets, private user data, or uploaded media to GitHub.
    """)
    st.markdown("**Model:** `yolov8n.pt` (small pretrained YOLO model; first run may download weights).")
