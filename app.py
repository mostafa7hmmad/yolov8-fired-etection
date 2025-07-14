import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from ultralytics import YOLO

# Load YOLOv8 model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# WebRTC config (needed for Streamlit Cloud to allow cam access)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Streamlit UI
st.title("🔥 Fire Detection via Facecam (Live)")
st.markdown("This app uses your webcam to detect **fire** in real-time.")

# Video transformer class
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        results = self.model.predict(source=img, imgsz=640, conf=0.3, verbose=False)
        
        for result in results:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                label = self.model.names[int(cls)]
                if label.lower() == "fire":
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(img, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 0, 255), 2)
        return img

# Run the webcam stream
webrtc_streamer(
    key="fire-cam",
    mode="sendrecv",
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
