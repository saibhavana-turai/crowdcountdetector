# app.py

import streamlit as st
import cv2
import numpy as np
import time
import os
import requests
import queue
import av

from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from streamlit_webrtc import webrtc_streamer, WebRtcMode

# ================= PATHS =================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)

CSR_A = os.path.join(MODEL_DIR, "csrnet_a.pth")
CSR_B = os.path.join(MODEL_DIR, "csrnet_b.pth")

CSR_A_URL = "https://huggingface.co/saibhavana-turai/crowd-counting-csrnet/resolve/main/csrnet_best_part_a.pth?download=true"
CSR_B_URL = "https://huggingface.co/saibhavana-turai/crowd-counting-csrnet/resolve/main/csrnet_best_part_b.pth?download=true"

# ================= DATABASE =================

Base = declarative_base()

class User(Base):
    __tablename__ = "user"
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True)
    password = Column(String)

engine = create_engine(
    f"sqlite:///{os.path.join(INSTANCE_DIR, 'users.db')}",
    connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# ================= ALERT =================

from alert_system import send_alert

# ================= MODEL DOWNLOAD =================

def is_valid_pytorch_file(path):
    """Check PyTorch binary magic header"""
    try:
        with open(path, "rb") as f:
            return f.read(2) == b"\x80\x04"  # pickle header
    except:
        return False

def download_model(url, path):
    if os.path.exists(path) and is_valid_pytorch_file(path):
        return

    if os.path.exists(path):
        os.remove(path)

    st.info(f"Downloading {os.path.basename(path)}")
    r = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)

    if not is_valid_pytorch_file(path):
        st.error("Downloaded file is NOT a valid PyTorch model")
        st.stop()

def ensure_models():
    download_model(CSR_A_URL, CSR_A)
    download_model(CSR_B_URL, CSR_B)

# ================= CSRNET LOADER (NO CACHE) =================

def load_csrnet_safe(path):
    import torch
    from torchvision import models

    class CSRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.frontend = torch.nn.Sequential(*list(vgg.features.children())[:23])
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
            self.output = torch.nn.Conv2d(64, 1, 1)

        def forward(self, x):
            return self.output(self.backend(self.frontend(x)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=True
    )

    model_dict = model.state_dict()
    filtered = {
        k.replace("module.", ""): v
        for k, v in checkpoint.items()
        if k.replace("module.", "") in model_dict
        and model_dict[k.replace("module.", "")].shape == v.shape
    }

    model_dict.update(filtered)
    model.load_state_dict(model_dict, strict=False)
    model.eval()
    return model

# ================= PROCESS =================

def preprocess(frame):
    import torch
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame = (frame - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()

def count_people(frame, model, user, threshold):
    import torch
    with torch.no_grad():
        density = model(preprocess(frame))[0, 0].cpu().numpy()

    count = int(max(density.sum(), 0))

    if user and count >= threshold:
        if time.time() - st.session_state.last_alert > 15:
            send_alert(count, user["email"])
            st.session_state.last_alert = time.time()

    heat = cv2.applyColorMap(
        cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
    out = cv2.addWeighted(frame, 0.6, heat, 0.4, 0)
    cv2.putText(out, f"Count: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
    return out

# ================= DASHBOARD =================

def dashboard():
    st.set_page_config(layout="wide", page_title="CrowdSense")

    ensure_models()
    model = load_csrnet_safe(CSR_A)

    if "last_alert" not in st.session_state:
        st.session_state.last_alert = 0

    threshold = st.sidebar.slider("Alert Threshold", 10, 200, 50)

    q = queue.Queue()

    def callback(frame):
        img = frame.to_ndarray(format="bgr24")
        q.put(img)
        return frame

    ctx = webrtc_streamer(
        key="cam",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
    )

    view = st.empty()

    while ctx.state.playing:
        try:
            frame = q.get(timeout=1)
        except queue.Empty:
            continue

        view.image(count_people(frame, model, st.session_state.user, threshold),
                   channels="BGR")

# ================= AUTH =================

def auth():
    db = SessionLocal()
    st.title("CrowdSense")

    mode = st.radio("Action", ["Login", "Register"])
    email = st.text_input("Email")
    pwd = st.text_input("Password", type="password")

    if st.button(mode):
        if mode == "Login":
            user = db.query(User).filter_by(email=email).first()
            if user and check_password_hash(user.password, pwd):
                st.session_state.logged_in = True
                st.session_state.user = {"email": email}
                st.rerun()
            else:
                st.error("Invalid credentials")
        else:
            try:
                db.add(User(email=email, password=generate_password_hash(pwd)))
                db.commit()
                st.success("Registered successfully")
            except IntegrityError:
                db.rollback()
                st.error("Email already exists")
    db.close()

# ================= ENTRY =================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    dashboard()
else:
    auth()
