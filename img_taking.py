import cv2
import os
import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator

name = 'Antonio'
i = 0
j = 0

path1 = os.path.join('Data', 'Train', name)
if not os.path.exists(path1):
    os.makedirs(path1)
path2 = os.path.join('Data', 'Test', name)
if not os.path.exists(path2):
    os.makedirs(path2)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
'haarcascade_frontalface_default.xml')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(150, 150)
    )
    for (x, y, w, h) in faces:
        face_img_gray = gray[y:y + h, x:x + w]
        laplacian_var = cv2.Laplacian(face_img_gray, cv2.CV_64F).var()
        if laplacian_var > 20 and j % 10 == 0:
            img_name = f'Frame_{i}.jpg'
        if i < 50:
            center = frame[110:610, 390:890]
            resized = cv2.resize(center, (250, 250))
            cv2.imwrite(os.path.join(path1, img_name), resized)
        if i >= 50:
            center = frame[110:610, 390:890]
            resized = cv2.resize(center, (250, 250))
            cv2.imwrite(os.path.join(path2, img_name), resized)
        if i > 63:
            cap.release()
            cv2.destroyAllWindows()
        i = i + 1
    j = j + 1
    frame_rec = cv2.rectangle(frame, (390, 110), (890, 610), (0, 0, 255), 2)
    cv2.imshow("Web cam", frame_rec)
    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows()

img_height, img_width = 250, 250
train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            brightness_range=(0.7, 1),
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=False,
            fill_mode='nearest')

# Pohranjuje informacije o generiranim slikama za trening set
n = 5
k = 0
total = len(os.listdir(path1))
for filename in os.listdir(path1):
    print("Step {} of {}".format(k + 1, total))
    image_path = os.path.join(path1, filename)
    image = keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width, 3))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    current_image_gen = train_datagen.flow(image,
                                           batch_size=1,
                                           save_to_dir=path1,
                                           save_prefix=filename[:len(filename) - 4],
                                           save_format="jpg")
    count = 0
    for image in current_image_gen:
        count += 1
        if count == n:
            break
    print('\tGenerate {} samples for file {}'.format(n, filename))
    k += 1
print("\nTotal number images generated = {}".format(n * total))

# Pohranjuje informacije o generiranim slikama za test set
n = 2
k2 = 0
total = len(os.listdir(path2))
for filename in os.listdir(path2):
    print("Step {} of {}".format(k2 + 1, total))
    image_path = os.path.join(path2, filename)
    image = keras.preprocessing.image.load_img(
        image_path, target_size=(img_height, img_width, 3))
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    current_image_gen = train_datagen.flow(image,
                                           batch_size=1,
                                           save_to_dir=path2,
                                           save_prefix=filename[:len(filename) - 4],
                                           save_format="jpg")
    count = 0
    for image in current_image_gen:
        count += 1
        if count == n:
            break
    print('\tGenerate {} samples for file {}'.format(n, filename))
    k2 += 1
print("\nTotal number images generated = {}".format(n * total))

