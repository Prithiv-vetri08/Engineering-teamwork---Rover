import cv2
import os
import numpy as np

def load_dataset(dataset_path):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_dict[current_label] = person_name
        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            faces_detected = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))
            if len(faces_detected) > 0:
                (x, y, w, h) = faces_detected[0]
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face = cv2.equalizeHist(face)
                faces.append(face)
                labels.append(current_label)
        current_label += 1

    return faces, np.array(labels), label_dict
