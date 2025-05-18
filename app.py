import cv2
import pygame
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("best.pt")

# Path to buzzer sound
BUZZER_SOUND_PATH = "1.mp3"

# Initialize the sound
pygame.mixer.init()
buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)

# Path to the video
VIDEO_PATH = "best.mp4"

# Read the video
cap = cv2.VideoCapture(VIDEO_PATH)

# لإبقاء الكشف لفترة مستمرة حتى لو لم يظهر في فريم معين
DETECTION_HOLD_FRAMES = 30
fire_detected_frames = 0
last_fire_boxes = []

# لحساب سرعة التشغيل (تقليل زمن الانتظار بين الفريمات)
normal_delay = 1  # تأخير افتراضي
speed_factor = 1.2
adjusted_delay = max(1, int(normal_delay / speed_factor))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to 700x400
    frame = cv2.resize(frame, (700, 400))

    fire_detected = False
    results = model.predict(source=frame, imgsz=640, conf=0.3, verbose=False)

    current_fire_boxes = []

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)
            if label.lower() == "fire":
                fire_detected = True
                fire_detected_frames = DETECTION_HOLD_FRAMES
                current_fire_boxes.append((x1, y1, x2, y2))

    if fire_detected:
        last_fire_boxes = current_fire_boxes
    else:
        if fire_detected_frames > 0:
            fire_detected_frames -= 1
            fire_detected = True

    if fire_detected_frames > 0:
        for x1, y1, x2, y2 in last_fire_boxes:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, "FIRE", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    if fire_detected:
        if not pygame.mixer.get_busy():
            buzzer.play()

    cv2.imshow("Fire Detection", frame)

    # السرعة الجديدة: delay محسوب
    if cv2.waitKey(adjusted_delay) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
