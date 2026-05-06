import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque

# =========================
# CONFIG
# =========================
MODEL_PATH = "best_densenet_final.keras"
FACE_MODEL_PATH = "blaze_face_short_range.tflite"

IMG_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

EMOTION_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']

pred_queue = deque(maxlen=5)

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("Model input:", model.input_shape)

# =========================
# MEDIAPIPE
# =========================
base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.6
)
detector = vision.FaceDetector.create_from_options(options)

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    H, W, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if not result.detections:
        pred_queue.clear()
    else:
        det = max(result.detections, key=lambda d: d.categories[0].score)

        bbox = det.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)

        # crop chuẩn hơn
        pad = 0.25
        x1 = max(0, int(x - w * pad))
        y1 = max(0, int(y - h * pad))
        x2 = min(W, int(x + w * (1 + pad)))
        y2 = min(H, int(y + h * (1 + pad)))

        face = rgb[y1:y2, x1:x2]

        if face.size == 0:
            continue

        # =====================
        # PREPROCESS ĐÚNG
        # =====================
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        # ❗ KHÔNG normalize nữa
        face = face.astype(np.float32)

        face = np.expand_dims(face, axis=0)

        # =====================
        # PREDICT
        # =====================
        pred = model(face, training=False)[0].numpy()

        pred_queue.append(pred)

        if len(pred_queue) < 3:
            continue

        avg_pred = np.mean(pred_queue, axis=0)

        idx = np.argmax(avg_pred)
        conf = avg_pred[idx]

        # DEBUG (quan trọng)
        print(f"{EMOTION_LABELS[idx]} - {conf:.2f}")

        # =====================
        # HIỂN THỊ
        # =====================
        if conf > CONFIDENCE_THRESHOLD:
            label = EMOTION_LABELS[idx]
            text = f"{label}: {conf*100:.1f}%"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0,255,0), 2)

    cv2.imshow("Emotion RAF-DB (Fixed)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()