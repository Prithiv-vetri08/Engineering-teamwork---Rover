import cv2
import os
import numpy as np
import time
import pyttsx3
import pickle
import threading
import queue

frame_count = 0
detect_every_n_frames = 5

model_path = "trained_model.yml"
label_dict_path = "label_dict.pkl"

# Path to dataset folder
dataset_path = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project"

# === TTS SETUP ===
speak_queue = queue.Queue()

def tts_worker():
    engine = pyttsx3.init()
    while True:
        name = speak_queue.get()
        if name is None:
            break  # Exit on sentinel
        engine.say(name)
        engine.runAndWait()
        speak_queue.task_done()

tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

def get_images_and_labels(path):
    faces = []
    labels = []
    label_dict = {}
    current_label = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for person_name in os.listdir(path):
        person_path = os.path.join(path, person_name)
        if not os.path.isdir(person_path):
            continue

        label_dict[current_label] = person_name
        print(f"Processing {person_name}...")

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            detected_faces = face_cascade.detectMultiScale(
                img, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

            if len(detected_faces) > 0:
                (x, y, w, h) = detected_faces[0]
                face = img[y:y+h, x:x+w]
                face = cv2.resize(face, (200, 200))
                face = cv2.equalizeHist(face)
                faces.append(face)
                labels.append(current_label)

        current_label += 1

    return faces, np.array(labels), label_dict

# Load dataset
print("Loading training data...")
if os.path.exists(model_path) and os.path.exists(label_dict_path):
    print("Loading existing trained model...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)

    with open(label_dict_path, "rb") as f:
        label_dict = pickle.load(f)
else:
    print("Training model for the first time...")
    faces, labels, label_dict = get_images_and_labels(dataset_path)

    if len(faces) == 0:
        print("ERROR: No faces found in training data!")
        exit()

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.save(model_path)

    with open(label_dict_path, "wb") as f:
        pickle.dump(label_dict, f)

# Initialize camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
cap.set(cv2.CAP_PROP_FPS, 60)

# Load face cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detection tuning
scaleFactor = 1.05
minNeighbors = 3
minSize = (50, 50)

# TTS cooldown
last_spoken = ""
last_spoken_time = 0
speak_cooldown = 3  # seconds

# Main loop
print("Starting recognition. Press 'q' to quit...")
frame_count = 0
detect_interval = 5
last_detection = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture error")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    frame_count += 1

    if frame_count % detect_interval == 0:
        small_gray = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)
        detected_faces = face_cascade.detectMultiScale(
            small_gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=(int(minSize[0] * 0.5), int(minSize[1] * 0.5))
        )
        if len(detected_faces) > 0:
            (x, y, w, h) = detected_faces[0]
            x, y, w, h = [int(val * 2) for val in (x, y, w, h)]
            last_detection = (x, y, w, h)

    if last_detection is not None:
        x, y, w, h = last_detection
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            face_roi = cv2.resize(roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)

            label, confidence = recognizer.predict(face_roi)

            if confidence < 70:
                name = label_dict.get(label, "Unknown")
                color = (0, 255, 0)
                if name != last_spoken or time.time() - last_spoken_time > speak_cooldown:
                    speak_queue.put(name)
                    last_spoken = name
                    last_spoken_time = time.time()
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Gracefully stop the TTS thread
speak_queue.put(None)
tts_thread.join()
