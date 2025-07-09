import cv2
import os
import numpy as np
import time

# Path to dataset folder
dataset_path = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\Project"

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
                
            # More sensitive face detection for training images
            detected_faces = face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(30, 30))
            
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
faces, labels, label_dict = get_images_and_labels(dataset_path)

if len(faces) == 0:
    print("ERROR: No faces found in training data!")
    exit()

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)
recognizer.save("trained_model.yml")

# Initialize camera
cap = cv2.VideoCapture(2)  # Try 0, 1, or 2
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Optimized detection parameters
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
scaleFactor = 1.05  # More sensitive (lower number)
minNeighbors = 3    # Fewer neighbors = more detections
minSize = (100, 100)  # Smaller minimum size

# Stabilization variables
last_detection = None
last_detection_time = 0
stabilization_threshold = 0.1  # seconds

print("Starting recognition. Press 'q' to quit...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame capture error")
        break
        
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    current_time = time.time()
    
    # Only detect if no recent detection or after stabilization period
    if last_detection is None or (current_time - last_detection_time) > stabilization_threshold:
        faces_detected = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize
        )
        
        if len(faces_detected) > 0:
            last_detection = faces_detected[0]
            last_detection_time = current_time
    
    # Use last detection if within stabilization period
    if last_detection is not None:
        (x, y, w, h) = last_detection
        
        # Verify the detection is still valid
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:  # Check we have valid image data
            face_roi = cv2.resize(roi, (200, 200))
            face_roi = cv2.equalizeHist(face_roi)
            
            label, confidence = recognizer.predict(face_roi)
            
            color = (0, 255, 0) if confidence < 70 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            name = label_dict.get(label, "Unknown") if confidence < 70 else "Unknown"
            text = f"{name} ({confidence:.1f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        else:
            last_detection = None
    
    cv2.imshow('Face Recognition', frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
