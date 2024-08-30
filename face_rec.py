import numpy as np
import tensorflow as tf
from tensorflow import keras
import cv2

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

color = RED

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model_name = 'face_classifier.h5'
face_classifier = keras.models.load_model(f'save/{model_name}')

class_names = ['Andrea', 'Antonio', 'Mia']

# Funkcija za proširenje slike lica
def get_extended_image(img, x, y, w, h, k=0.1):
    start_x = max(int(x - k * w), 0)
    start_y = max(int(y - k * h), 0)
    end_x = int(x + (1 + k) * w)
    end_y = int(y + (1 + k) * h)

    face_image = img[start_y:end_y, start_x:end_x]
    face_image = tf.image.resize(face_image, [250, 250])
    face_image = np.expand_dims(face_image, axis=0)
    return face_image


# Inicijalizacija video zapisa
video_capture = cv2.VideoCapture(0)
if not video_capture.isOpened():
    print("Nije moguće pristupiti kameri")
else:
    print("Streaming započeo - za izlaz pritisnite ESC")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Ne mogu primiti okvir (kraj strima?). Izlazim ...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
    for (x, y, w, h) in faces:
        face_image = get_extended_image(frame, x, y, w, h, 0.5)
        result = face_classifier.predict(face_image)
        prediction = class_names[np.array(result[0]).argmax(axis=0)]
        confidence = np.array(result[0]).max(axis=0)
        print(prediction, result, confidence)

        if prediction == class_names[0]:
            color = BLUE
        elif prediction == class_names[1]:
            color = GREEN
        elif prediction == class_names[2]:
            color = WHITE

        if confidence > 0.8:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "{:6} : {:.2f}%".format(prediction, confidence * 100), (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), RED, 2)

    cv2.imshow("Face detector - za izlaz pritisnite ESC", frame)

    key = cv2.waitKey(1)
    if key % 256 == 27:  # ESC
        break

video_capture.release()
cv2.destroyAllWindows()
print("Streaming završio")
