from dataset_loader import load_dataset
from trainer import train_recognizer
from recognizer import start_recognition
import os
import cv2
import pickle

# Change to your dataset path
DATASET_PATH = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project"
MODEL_PATH = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project\trained_model.yml"
LABELS_PATH = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project\labels.pkl"
CAMERA_INDEX = 1  # Change if needed

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        # ✅ Load trained model and labels
        print("Loading trained model...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)

        with open(LABELS_PATH, "rb") as f:
            label_dict = pickle.load(f)

        start_recognition(recognizer, label_dict, camera_index=CAMERA_INDEX)

    else:
        # ✅ Train model if no trained data found
        print("No trained model found. Training...")
        faces, labels, label_dict = load_dataset(DATASET_PATH)

        if len(faces) == 0:
            print("ERROR: No faces found in training data!")
            exit()

        recognizer = train_recognizer(faces, labels, MODEL_PATH)

        # Save labels for future use
        with open(LABELS_PATH, "wb") as f:
            pickle.dump(label_dict, f)

        # ✅ Start recognition immediately after training
        start_recognition(recognizer, label_dict, camera_index=CAMERA_INDEX)
