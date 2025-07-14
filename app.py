import streamlit as st
import cv2
import numpy as np
import os
from ultralytics import YOLO

# Constants
MODEL_PATH = "best.pt"

st.title("🔥 Fire Detection App (Cloud Ready)")

# Sidebar for mode
mode = st.sidebar.selectbox("Select Mode", ["Upload Image"])

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model file '{MODEL_PATH}' not found! Make sure you upload it with the app.")
    st.stop()

# Load YOLO model
model = YOLO(MODEL_PATH)

def detect_fire(image):
    fire_detected = False
    results = model.predict(source=image, imgsz=640, conf=0.3, verbose=False)
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            label = model.names[int(cls)]
            if label.lower() == "fire":
                fire_detected = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(image, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)
    return image, fire_detected


# --- Mode: Upload Image ---
if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (700, 400))

        st.info("Running fire detection... 🔍")
        processed_img, fire_found = detect_fire(img)

        if fire_found:
            st.error("🔥 FIRE DETECTED!")
        else:
            st.success("✅ No fire detected.")

        st.image(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB),
                 caption="Processed Image", use_column_width=True)
