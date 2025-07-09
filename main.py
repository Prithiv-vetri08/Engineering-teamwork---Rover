from dataset_loader import load_dataset
from trainer import train_recognizer
from recognizer import start_recognition
import os

# Change to your dataset path
DATASET_PATH = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project"
MODEL_PATH = "trained_model.yml"
CAMERA_INDEX = 2  # Change if needed

if __name__ == "__main__":
    print("Loading training data...")
    faces, labels, label_dict = load_dataset(DATASET_PATH)

    if len(faces) == 0:
        print("ERROR: No faces found in training data!")
        exit()

    recognizer = train_recognizer(faces, labels, MODEL_PATH)
    start_recognition(recognizer, label_dict, camera_index=CAMERA_INDEX)
