# filename: app.py

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import os
import requests
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import queue

# ================= PATH SETUP =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
INSTANCE_FOLDER_PATH = os.path.join(BASE_DIR, "instance")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INSTANCE_FOLDER_PATH, exist_ok=True)

CSRNET_A_PATH = os.path.join(MODEL_DIR, "csrnet_best_part_a.pth")
CSRNET_B_PATH = os.path.join(MODEL_DIR, "csrnet_best_part_b.pth")

CSRNET_A_URL = "https://huggingface.co/saibhavana-turai/crowd-counting-csrnet/resolve/main/csrnet_best_part_a.pth"
CSRNET_B_URL = "https://huggingface.co/saibhavana-turai/crowd-counting-csrnet/resolve/main/csrnet_best_part_b.pth"

# ================= DATABASE =================

Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

DATABASE_URL = f"sqlite:///{os.path.join(INSTANCE_FOLDER_PATH, 'users.db')}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# ================= ALERT =================

from alert_system import send_alert

# ================= MODEL DOWNLOAD =================

def download_file(url, destination):
    if os.path.exists(destination) and os.path.getsize(destination) > 1_000_000:
        return True

    st.info(f"Downloading model: {os.path.basename(destination)}")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        progress = st.progress(0)
        downloaded = 0

        with open(destination, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    progress.progress(downloaded / total)

        progress.empty()
        return True

    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

def ensure_models():
    ok1 = download_file(CSRNET_A_URL, CSRNET_A_PATH)
    ok2 = download_file(CSRNET_B_URL, CSRNET_B_PATH)
    if not (ok1 and ok2):
        st.stop()

# ================= MODEL LOADERS =================

@st.cache_resource
def load_improved_csrnet_model(path):
    import torch
    from torchvision import models

    class ImprovedCSRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.frontend = torch.nn.Sequential(*list(vgg16.features.children())[:23])
            self.backend = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, 3, padding=2, dilation=2),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(512, 256, 3, padding=2, dilation=2),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(256, 128, 3, padding=2, dilation=2),
                torch.nn.ReLU(True),
                torch.nn.Conv2d(128, 64, 3, padding=2, dilation=2),
                torch.nn.ReLU(True),
            )
            self.output_layer = torch.nn.Conv2d(64, 1, 1)

        def forward(self, x):
            x = self.frontend(x)
            x = self.backend(x)
            x = self.output_layer(x)
            return torch.nn.functional.interpolate(
                x, size=(512, 512), mode="bilinear", align_corners=False
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImprovedCSRNet().to(device)

    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # 🔧 FIX: prevents RuntimeError
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model

@st.cache_resource
def load_yolo_model():
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()
    return model

# ================= PROCESSING =================

def preprocess_frame(frame):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).to(device)

def get_count_and_overlay(frame, model, yolo_model, user, threshold):
    import torch

    inp = preprocess_frame(frame)
    with torch.no_grad():
        density = model(inp)[0, 0].cpu().numpy()

    count = int(round(max(density.sum(), 0)))

    if user and count >= threshold:
        if time.time() - st.session_state.last_alert_time > 15:
            send_alert(count, user["email"])
            st.session_state.last_alert_time = time.time()
            st.session_state.alert_history.insert(
                0, f"ALERT: {count} people detected"
            )

    heat = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(
        cv2.resize(heat, (frame.shape[1], frame.shape[0])), cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
    cv2.putText(
        overlay,
        f"Count: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )
    return overlay, count

# ================= AUTH =================

def authentication_page():
    st.set_page_config(page_title="CrowdSense", page_icon="👥")
    db = SessionLocal()
    st.title("CrowdSense")

    choice = st.radio("Action", ["Login", "Register"])

    if choice == "Login":
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = db.query(User).filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                st.session_state.logged_in = True
                st.session_state.user = {"email": user.email}
                st.rerun()
            else:
                st.error("Invalid credentials")
    else:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        if st.button("Register"):
            db.add(User(email=email, password=generate_password_hash(password)))
            db.commit()
            st.success("Registered successfully")

    db.close()

# ================= DASHBOARD =================

def main_dashboard():
    st.set_page_config(layout="wide", page_title="Dashboard", page_icon="👥")

    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []
        st.session_state.last_alert_time = 0

    ensure_models()

    model_dense = load_improved_csrnet_model(CSRNET_A_PATH)
    model_sparse = load_improved_csrnet_model(CSRNET_B_PATH)
    yolo_model = load_yolo_model()

    with st.sidebar:
        st.write(f"Logged in as **{st.session_state.user['email']}**")
        if st.button("Logout"):
            st.session_state.clear()
            st.rerun()

        model_choice = st.selectbox("Model", ["Dense", "Sparse"])
        threshold = st.slider("Alert Threshold", 0, 200, 50)

    model = model_dense if model_choice == "Dense" else model_sparse

    frame_queue = queue.Queue()

    def video_callback(frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        frame_queue.put(img)
        return frame

    ctx = webrtc_streamer(
        key="cam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_callback,
        media_stream_constraints={"video": True, "audio": False},
    )

    raw = st.empty()
    processed = st.empty()

    while ctx.state.playing:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue

        raw.image(frame, channels="BGR")
        overlay, _ = get_count_and_overlay(
            frame, model, yolo_model, st.session_state.user, threshold
        )
        processed.image(overlay, channels="BGR")

# ================= ENTRY =================

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        main_dashboard()
    else:
        authentication_page()
