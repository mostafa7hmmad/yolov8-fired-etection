# app.py
import os
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import gdown
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Constants
MODEL_PATH = "best.pt"
BUZZER_SOUND_PATH = "1.mp3"
DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1N7PmIx1hHBlXtYNvi0u2dYK18n5wWtPA"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    st.info("Downloading YOLO model from Google Drive...")
    gdown.download_folder(DRIVE_FOLDER_URL, output="./", quiet=False, use_cookies=False)

# RTC configuration for WebRTC
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        pygame.mixer.init()
        self.buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(source=img, imgsz=640, conf=0.3, verbose=False)
        fire_detected = False
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                label = self.model.names[int(cls)]
                if label.lower() == "fire":
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if fire_detected:
            self.buzzer.play()
        return img

# Streamlit UI
st.title("Fire Detection App 🚒🔥")
app_mode = st.sidebar.selectbox("Mode", ["Upload Image", "Webcam"])

if app_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (700, 400))

        model = YOLO(MODEL_PATH)
        pygame.mixer.init()
        buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)

        results = model.predict(source=image, imgsz=640, conf=0.3, verbose=False)
        fire_detected = False
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                label = model.names[int(cls)]
                if label.lower() == "fire":
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if fire_detected:
            buzzer.play()

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Processed Image", use_column_width=True)

else:
    webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )

