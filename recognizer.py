import cv2
import time

def start_recognition(recognizer, label_dict, camera_index=0, model_path="trained_model.yml"):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    scaleFactor = 1.05
    minNeighbors = 3
    minSize = (100, 100)

    last_detection = None
    last_detection_time = 0
    stabilization_threshold = 0.1

    print("Starting recognition. Press 'q' to quit...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture error")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        current_time = time.time()

        if last_detection is None or (current_time - last_detection_time) > stabilization_threshold:
            faces_detected = face_cascade.detectMultiScale(
                gray, scaleFactor=scaleFactor,
                minNeighbors=minNeighbors, minSize=minSize
            )

            if len(faces_detected) > 0:
                last_detection = faces_detected[0]
                last_detection_time = current_time

        if last_detection is not None:
            (x, y, w, h) = last_detection
            roi = gray[y:y+h, x:x+w]
            if roi.size > 0:
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
