import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# ==========================================
# LOAD MODEL
# ==========================================
print("Đang nạp mô hình AI...")
#model_path = "best_model2.h5"
model_path = "best_custom_cnn_emotion.h5"

model = tf.keras.models.load_model(model_path)

emotion_labels = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

# ==========================================
# LOAD FACE CASCADE
# ==========================================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# ==========================================
# CAMERA
# ==========================================
cap = cv2.VideoCapture(0)

print("Sẵn sàng! Nhấn 'q' để thoát.")

# ==========================================
# LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.25,
        minNeighbors=10,
        minSize=(65, 65)
    )

    for (x, y, w, h) in faces:

        # ROI RGB
        roi_color = frame[y:y+h, x:x+w]
        roi_rgb = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

        # Resize 96x96
        roi_resized = cv2.resize(roi_rgb, (100, 100))

        roi_normalized = roi_resized / 255.0
        roi_expanded = np.expand_dims(roi_normalized, axis=0)

        # Predict
        pred = model(roi_expanded, training=False)
        pred_array = pred.numpy()[0]

        max_index = np.argmax(pred_array)
        confidence = pred_array[max_index]

        #Confidence threshold
        if confidence < 0.4:
            continue
        
        label = emotion_labels[max_index]
        text = f"{label}: {confidence*100:.1f}%"

        COLOR_MAP = {
            "Angry": (0, 0, 255),      # đỏ
            "Fear": (0, 0, 255),       # đỏ
            "Disgust": (0, 0, 255),    # đỏ
            "Sad": (255, 0, 0),        # xanh dương
            "Surprise": (0, 255, 255), # vàng
            "Happy": (0, 255, 0),      # xanh lá
            "Neutral": (0, 255, 0)     # xanh lá
        }

        color = COLOR_MAP.get(label, (0, 255, 0))  # default xanh lá

        # Draw
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, cv2.FILLED)

        cv2.putText(frame, text, (x+5, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    cv2.imshow("Emotion Recognition - Stable", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()