import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Constants
MODEL_PATH = "best.pt"
BUZZER_SOUND_PATH = "1.mp3"

# RTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file not found at: {MODEL_PATH}")
    st.stop()

# Try to initialize pygame mixer (handle Streamlit Cloud error)
try:
    pygame.mixer.init()
    buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)
except Exception as e:
    buzzer = None
    st.warning("⚠️ Audio is not supported in this environment. Buzzer disabled.")

# Streamlit title
st.title("🔥 Fire Detection App")
app_mode = st.sidebar.selectbox("Choose Mode", ["Upload Image", "Webcam"])


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        try:
            pygame.mixer.init()
            self.buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)
        except Exception:
            self.buzzer = None

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
                    cv2.putText(img, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

        if fire_detected and self.buzzer:
            self.buzzer.play()
        return img


if app_mode == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (700, 400))

        model = YOLO(MODEL_PATH)
        results = model.predict(source=image, imgsz=640, conf=0.3, verbose=False)

        fire_detected = False
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                label = model.names[int(cls)]
                if label.lower() == "fire":
                    fire_detected = True
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(image, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)

        if fire_detected and buzzer:
            buzzer.play()

        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                 caption="Processed Image", use_column_width=True)

else:
    webrtc_streamer(
        key="fire-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
    )
