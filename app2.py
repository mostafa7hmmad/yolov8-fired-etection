import cv2
import pygame
from ultralytics import YOLO

# تحميل الموديل (غير الاسم لو عندك موديل مدرب خاص بك)
model = YOLO("best.pt")  # أو مسار fire_detector.pt بتاعك

# مسار ملف صوت البازر
BUZZER_SOUND_PATH = "1.mp3"

# تهيئة pygame لتشغيل الصوت
pygame.mixer.init()
buzzer = pygame.mixer.Sound(BUZZER_SOUND_PATH)

# فتح الكاميرا (0 معناها الكاميرا الأساسية)
cap = cv2.VideoCapture(0)

# لو الكاميرا مش شغالة
if not cap.isOpened():
    print("❌ لا يمكن فتح الكاميرا")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ فشل في قراءة الكاميرا")
        break

    # استخدم الموديل للكشف
    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False)

    fire_detected = False

    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, box)

            if label.lower() == "fire":  # عدّله لو اسم الكلاس مختلف
                fire_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "FIRE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # لو كشف حريق يشغل صوت البازر
    if fire_detected:
        if not pygame.mixer.get_busy():
            buzzer.play()

    # عرض الفريم
    cv2.imshow("🔥 Fire Detection - Webcam", frame)

    # اضغط "q" للخروج
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# تنظيف
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
