import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import os

# Constants
MODEL_PATH = "best.pt"

# Streamlit Title
st.title("🔥 Real-time Fire Detection via Facecam")

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found. Make sure it's uploaded.")
    st.stop()

# Load YOLOv8 model
model = YOLO(MODEL_PATH)

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Define class to process webcam frames
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.model = model

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Fire detection
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

# Display webcam stream using streamlit-webrtc
webrtc_streamer(
    key="fire-cam",
    mode=WebRtcMode.SENDRECV,  # ✅ Fix: use enum, not string
    rtc_configuration=RTC_CONFIGURATION,
    video_transformer_factory=VideoTransformer,
    media_stream_constraints={"video": True, "audio": False},
)
