import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import pygame
import os

st.title("🔥 Fire Detection with Live Face Cam")

MODEL_PATH = "best.pt"
BUZZER_SOUND_PATH = "1.mp3"

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found!")
    st.stop()

model = YOLO(MODEL_PATH)

# Try to init audio
try:
    pygame.mixer.init()
    buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)
except:
    buzzer = None
    st.warning("Audio device not available. Buzzer disabled.")

# Toggle webcam mode
run = st.checkbox('Start Webcam')

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # Use 0 for default webcam

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Could not access webcam.")
        break

    frame = cv2.resize(frame, (640, 480))

    # Fire detection
    results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)
    fire_detected = False

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            label = model.names[int(cls)]
            if label.lower() == "fire":
                fire_detected = True
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.9, (0, 0, 255), 2)

    if fire_detected and buzzer:
        buzzer.play()

    # Show image
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# Release webcam after stop
cap.release()
