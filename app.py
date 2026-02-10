# filename: app.py

import streamlit as st
import cv2
import numpy as np
import time
import pandas as pd
import os
import requests
from werkzeug.security import generate_password_hash, check_password_hash

# --- Database Setup ---
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class User(Base):
    __tablename__ = 'user'
    id = Column(Integer, primary_key=True)
    email = Column(String(100), unique=True, nullable=False)
    password = Column(String(200), nullable=False)

INSTANCE_FOLDER_PATH = os.path.join(os.path.dirname(__file__), 'instance')
DATABASE_URL = f"sqlite:///{os.path.join(INSTANCE_FOLDER_PATH, 'users.db')}"
os.makedirs(INSTANCE_FOLDER_PATH, exist_ok=True)

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)

# --- Project Imports ---
from alert_system import send_alert

# --- Helper: Download Models ---
def download_file(url, destination):
    if not os.path.exists(destination):
        with st.spinner(f"Downloading {os.path.basename(destination)}..."):
            r = requests.get(url, stream=True)
            with open(destination, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
    return True

# --- Load Models ---
@st.cache_resource
def load_csrnet(path):
    import torch
    from torchvision import models

    class CSRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.frontend = torch.nn.Sequential(*list(vgg.features.children())[:23])
            self.backend = torch.nn.Sequential(
                torch.nn.Conv2d(512, 256, 3, padding=2, dilation=2),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 128, 3, padding=2, dilation=2),
                torch.nn.ReLU(inplace=True),
            )
            self.output = torch.nn.Conv2d(128, 1, 1)

        def forward(self, x):
            x = self.frontend(x)
            x = self.backend(x)
            return self.output(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CSRNet().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

@st.cache_resource
def load_yolo():
    import torch
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
    model.eval()
    return model

# --- Frame Processing ---
def preprocess(frame):
    import torch
    frame = cv2.resize(frame, (512, 512))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame / 255.0
    tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).float()
    return tensor

def predict(frame, csrnet, yolo, threshold, user):
    import torch
    tensor = preprocess(frame)
    with torch.no_grad():
        density = csrnet(tensor)[0, 0].numpy()
    count = int(density.sum())

    # YOLO fallback for sparse scenes
    if count < 2:
        results = yolo(frame[..., ::-1])
        count = int((results.pred[0][:, -1] == 0).sum())

    overlay = frame.copy()
    cv2.putText(
        overlay,
        f"Count: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    if count >= threshold and user:
        if time.time() - st.session_state.last_alert_time > 15:
            send_alert(count, user["email"])
            st.session_state.last_alert_time = time.time()
            st.session_state.alert_history.insert(
                0, f"ALERT: {count} people detected"
            )

    return overlay, count

# --- Auth Page ---
def auth_page():
    st.title("CrowdSense Login")
    db = SessionLocal()

    choice = st.radio("Action", ["Login", "Register"])
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button(choice):
        if choice == "Login":
            user = db.query(User).filter_by(email=email).first()
            if user and check_password_hash(user.password, password):
                st.session_state.logged_in = True
                st.session_state.user = {"email": user.email}
                st.rerun()
            else:
                st.error("Invalid credentials")
        else:
            if db.query(User).filter_by(email=email).first():
                st.error("User already exists")
            else:
                db.add(
                    User(
                        email=email,
                        password=generate_password_hash(password),
                    )
                )
                db.commit()
                st.success("Registered successfully")

    db.close()

# --- Dashboard ---
def dashboard():
    st.title("👥 CrowdSense Dashboard")

    if "chart_data" not in st.session_state:
        st.session_state.chart_data = pd.DataFrame(columns=["Time", "Count"])
    if "alert_history" not in st.session_state:
        st.session_state.alert_history = []
    if "last_alert_time" not in st.session_state:
        st.session_state.last_alert_time = 0

    # Load models
    os.makedirs("models", exist_ok=True)
    download_file("MODEL_URL_A", "models/csrnet.pth")

    csrnet = load_csrnet("models/csrnet.pth")
    yolo = load_yolo()

    threshold = st.slider("Alert Threshold", 0, 200, 50)

    st.subheader("📷 Webcam Input")
    camera = st.camera_input("Click to enable camera")

    col1, col2 = st.columns(2)
    raw = col2.empty()
    processed = col1.empty()

    if camera:
        frame = cv2.imdecode(
            np.frombuffer(camera.getvalue(), np.uint8), cv2.IMREAD_COLOR
        )
        raw.image(frame, channels="BGR")

        overlay, count = predict(
            frame, csrnet, yolo, threshold, st.session_state.user
        )
        processed.image(overlay, channels="BGR")

        st.session_state.chart_data = pd.concat(
            [
                st.session_state.chart_data,
                pd.DataFrame(
                    {"Time": [time.strftime("%H:%M:%S")], "Count": [count]}
                ),
            ]
        ).tail(30)

    st.line_chart(st.session_state.chart_data.set_index("Time"))
