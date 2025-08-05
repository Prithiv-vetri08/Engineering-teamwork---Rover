from flask import Flask, render_template_string, Response
import cv2
import threading
import serial
import time

# === Arduino Setup ===
arduino = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)  # Update this if needed
time.sleep(2)

# === Camera Setup ===
cap = cv2.VideoCapture(0)
follow_line_flag = False

# === Flask HTML ===
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Rover Control with Vision</title>
</head>
<body style="background: black; color: white; text-align: center;">
    <h1>Vision-Based Rover Control</h1>
    <img src="/video" width="640" height="480"/>
    <br><br>
    <button onclick="fetch('/start', {method: 'POST'})">Start Line Following</button>
    <button onclick="fetch('/stop', {method: 'POST'})">Stop Rover</button>
</body>
</html>
"""

app = Flask(__name__)

@app.route("/")
def index():
    return render_template_string(HTML)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            continue
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
def start_line_following():
    global follow_line_flag
    follow_line_flag = True
    return "", 204

@app.route("/stop", methods=["POST"])
def stop_rover():
    global follow_line_flag
    follow_line_flag = False
    arduino.write(b'F')  # Send stop command
    return "", 204

def line_following():
    global follow_line_flag
    while True:
        if not follow_line_flag:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if not ret:
            continue

        height, width = frame.shape[:2]
        roi = frame[height//2:, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                center = width // 2
                error = cx - center

                if abs(error) < 30:
                    arduino.write(b'S')  # Forward
                elif error > 30:
                    arduino.write(b'R')  # Right
                else:
                    arduino.write(b'L')  # Left
            else:
                arduino.write(b'F')  # Stop
        else:
            arduino.write(b'F')  # Stop

# Start the vision thread
threading.Thread(target=line_following, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
