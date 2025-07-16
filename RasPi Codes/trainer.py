import cv2
import numpy as np

def train_recognizer(faces, labels, model_path="trained_model.yml"):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, labels)
    recognizer.save(model_path)
    print("Training complete. Model saved.")
    return recognizer
