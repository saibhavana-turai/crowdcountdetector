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
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# --- Project Imports ---
from alert_system import send_alert

# --- Helper function to download files ---
def download_file(url, destination):
    if not os.path.exists(destination):
        st.info(f"Downloading model: {os.path.basename(destination)}...")
        try:
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                progress_bar = st.progress(0)
                bytes_downloaded = 0
                with open(destination, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bytes_downloaded += len(chunk)
                        if total_size > 0:
                            progress_bar.progress(min(1.0, bytes_downloaded / total_size))
            progress_bar.empty()
        except requests.exceptions.RequestException as e:
            st.error(f"Error downloading model: {e}"); return False
    return True

# --- Model Loading (with Caching) - DEFINITIVE FIX ---
@st.cache_resource
def load_improved_csrnet_model(path):
    # --- FIX: Import dependencies LOCALLY inside the function ---
    import torch
    from torchvision import models

    # --- FIX: Define the class LOCALLY inside the function ---
    class ImprovedCSRNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.frontend = torch.nn.Sequential(*list(vgg16.features.children())[:23])
            self.backend = torch.nn.Sequential(
                torch.nn.Conv2d(512, 512, 3, padding=2, dilation=2), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(512, 512, 3, padding=2, dilation=2), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(512, 512, 3, padding=2, dilation=2), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(512, 256, 3, padding=2, dilation=2), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 128, 3, padding=2, dilation=2), torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(128, 64, 3, padding=2, dilation=2), torch.nn.ReLU(inplace=True),
            )
            self.output_layer = torch.nn.Conv2d(64, 1, 1)
        def forward(self, x):
            x = self.frontend(x); x = self.backend(x); x = self.output_layer(x)
            return torch.nn.functional.interpolate(x, size=(512, 512), mode='bilinear', align_corners=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ImprovedCSRNet().to(device)
    import torch.serialization
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict); model.eval()
    return model

@st.cache_resource
def load_yolo_model():
    # --- FIX: Import torch LOCALLY inside the function ---
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, force_reload=False).to(device)
    model.eval()
    return model

# --- Core Processing Logic ---
def preprocess_frame(frame):
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    IMG_SIZE=(512, 512); IMAGENET_MEAN=np.array([0.485, 0.456, 0.406]); IMAGENET_STD=np.array([0.229, 0.224, 0.225])
    frame_resized = cv2.resize(frame, (IMG_SIZE[1], IMG_SIZE[0])); img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    img = (img_rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

def get_count_and_overlay(frame, model, yolo_model, user, threshold):
    import torch
    input_tensor = preprocess_frame(frame)
    with torch.no_grad(): density_np = model(input_tensor)[0, 0].cpu().numpy()
    count = max(density_np.sum(), 0)
    if count < 1.7:
        def yolo_person_count(frame_for_yolo, yolo):
            results = yolo(frame_for_yolo[..., ::-1]); pred = results.pred[0]
            return int((pred[:, -1].cpu().numpy() == 0).sum()) if pred is not None and len(pred) > 0 else 0
        yolo_count = yolo_person_count(frame, yolo_model)
        if yolo_count > count: count = float(yolo_count)
    final_count = max(0, int(round(count)))
    if final_count >= threshold and user and 'last_alert_time' in st.session_state:
        if (time.time() - st.session_state.last_alert_time) > 15.0:
            send_alert(final_count, user['email']); st.session_state.last_alert_time = time.time()
            st.session_state.alert_history.insert(0, f"ALERT: Count of {final_count} detected at {time.strftime('%H:%M:%S')}")
    dmap_resized = cv2.resize(density_np, (frame.shape[1], frame.shape[0])); dmap_normalized = (dmap_resized / (dmap_resized.max() + 1e-8) * 255).astype(np.uint8)
    density_color = cv2.applyColorMap(dmap_normalized, cv2.COLORMAP_JET); overlay = cv2.addWeighted(frame, 0.6, density_color, 0.4, 0)
    cv2.putText(overlay, f'Predicted Count: {final_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return overlay, final_count

# --- Authentication UI ---
def authentication_page():
    st.set_page_config(layout="centered", page_icon="👥", page_title="Welcome")
    if 'auth_view' not in st.session_state: st.session_state.auth_view = "Login"
    _, col2, _ = st.columns([1, 2, 1])
    with col2:
        st.title("Welcome to CrowdSense")
        choice = st.radio("Action", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
        st.session_state.auth_view = choice; db_session = SessionLocal()
        if st.session_state.auth_view == "Login":
            st.subheader("Login to your account")
            with st.form("login_form"):
                email = st.text_input("Email"); password = st.text_input("Password", type="password")
                if st.form_submit_button("Login"):
                    user = db_session.query(User).filter_by(email=email).first()
                    if user and check_password_hash(user.password, password):
                        st.session_state.logged_in = True; st.session_state.user = {'email': user.email, 'id': user.id}; st.rerun()
                    else: st.error("Invalid email or password")
        else:
            st.subheader("Create a new account")
            with st.form("register_form"):
                email = st.text_input("Email"); password = st.text_input("Password", type="password")
                if st.form_submit_button("Register"):
                    if db_session.query(User).filter_by(email=email).first(): st.error("Email already exists.")
                    else:
                        new_user = User(email=email, password=generate_password_hash(password, method='pbkdf2:sha256'))
                        db_session.add(new_user); db_session.commit(); st.success("Registration successful! Please login."); st.balloons()
                        st.session_state.auth_view = "Login"; time.sleep(2); st.rerun()
        db_session.close()

# --- Main Application Dashboard ---
def main_dashboard():
    st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="👥", page_title="Dashboard")
    if 'chart_data' not in st.session_state: st.session_state.chart_data = pd.DataFrame(columns=['Time', 'Count'])
    if 'alert_history' not in st.session_state: st.session_state.alert_history = []
    if 'last_alert_time' not in st.session_state: st.session_state.last_alert_time = 0
    
    MODELS_DIR = "models"
    os.makedirs(MODELS_DIR, exist_ok=True)
    MODEL_URL_A = "https://drive.google.com/uc?export=download&id=1JIwn7tWrtDF7WfEpVR6XOBhINQpQBlA3"
    MODEL_URL_B = "https://drive.google.com/uc?export=download&id=1ivhWD8LYVOEVkZMBqi-p6Rcj2Bjn3ifm"
    MODEL_PATH_A = os.path.join(MODELS_DIR, "csrnet_best_part_a.pth")
    MODEL_PATH_B = os.path.join(MODELS_DIR, "csrnet_best_part_b.pth")
    if not (download_file(MODEL_URL_A, MODEL_PATH_A) and download_file(MODEL_URL_B, MODEL_PATH_B)):
        st.error("Model download failed. App cannot continue."); return

    model_dense = load_improved_csrnet_model(MODEL_PATH_A)
    model_sparse = load_improved_csrnet_model(MODEL_PATH_B)
    yolo_model = load_yolo_model()
    
    with st.sidebar:
        st.title("CrowdSense"); st.markdown("---")
        user_info = st.session_state.get('user', {}); st.write(f"Logged in as: **{user_info.get('email', 'N/A')}**")
        if st.button("Logout"):
            for key in st.session_state.keys(): del st.session_state[key]
            st.rerun()
        st.markdown("---"); st.header("Controls")
        model_choice = st.selectbox("Analysis Model", ["Dense Crowd Model", "Sparse Crowd Model"])
        current_model = model_dense if "Dense" in model_choice else model_sparse
        threshold = st.slider("Alert Threshold", 0, 200, 50)
        st.header("Input Source")
        use_webcam = st.button("Use Webcam")
        video_file = st.file_uploader("Upload Video", type=['mp4', 'mov', 'avi'])
        image_file = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'])
    
    st.title("Live Analysis Dashboard"); status_placeholder = st.empty()
    col1, col2 = st.columns(2);
    with col1: st.header("Processed Feed / Heatmap"); processed_feed = st.empty()
    with col2: st.header("Raw Input Feed"); raw_feed = st.empty()
    st.markdown("---"); col3, col4 = st.columns([2, 1])
    with col3: st.header("Live Crowd Count Trend"); chart_placeholder = st.empty()
    with col4: st.header("Alert History"); alert_placeholder = st.expander("Show/Hide Alerts", expanded=True)
    
    cap = None;
    if use_webcam: cap = cv2.VideoCapture(0); status_placeholder.info("Processing webcam feed...")
    elif video_file:
        with open("temp_video.mp4", "wb") as f: f.write(video_file.getbuffer())
        cap = cv2.VideoCapture("temp_video.mp4"); status_placeholder.info(f"Processing uploaded video: {video_file.name}")
    elif image_file:
        bytes_data = image_file.getvalue(); cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        raw_feed.image(cv2_img, channels="BGR")
        overlay, count = get_count_and_overlay(cv2_img, current_model, yolo_model, user_info, threshold)
        processed_feed.image(overlay, channels="BGR"); status_placeholder.success(f"Image processed. Predicted Count: {count}")
    else: status_placeholder.info("Select an input source from the sidebar to begin.")
    
    if cap:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: status_placeholder.warning("Video ended or failed to read frame."); break
            raw_feed.image(frame, channels="BGR")
            overlay, count = get_count_and_overlay(frame, current_model, yolo_model, user_info, threshold)
            processed_feed.image(overlay, channels="BGR")
            new_data = pd.DataFrame({'Time': [time.strftime('%H:%M:%S')], 'Count': [count]})
            st.session_state.chart_data = pd.concat([st.session_state.chart_data, new_data]).tail(30)
            with chart_placeholder: st.line_chart(st.session_state.chart_data.set_index('Time'))
            with alert_placeholder:
                for alert in st.session_state.alert_history: st.warning(alert)
            time.sleep(0.01)
        cap.release()

# --- Main App Router ---
if __name__ == '__main__':
    if 'logged_in' not in st.session_state: st.session_state.logged_in = False
    
    if st.session_state.logged_in:
        main_dashboard()
    else:
        authentication_page()