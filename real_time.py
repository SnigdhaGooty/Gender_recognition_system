import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf

# Load models
face_model = YOLO(r"D:\NEW FOLDER\Gender_recognition\yolov8n-face-lindevs.pt")
gender_model = tf.keras.models.load_model(r"D:\NEW FOLDER\Gender_recognition\gender_model.keras")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = face_model(frame)[0]

    for box in results.boxes:
        x1,y1,x2,y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]

        if face.size == 0:
            continue

        face_input = cv2.resize(face, (128,128))
        face_input = face_input / 255.0
        face_input = np.expand_dims(face_input, axis=0)

        pred = gender_model.predict(face_input)[0][0]
        label = "Male" if pred > 0.5 else "Female"

        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(frame,label,(x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

    cv2.imshow("Gender Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
