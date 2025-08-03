from dataset_loader import load_dataset
from trainer import train_recognizer
from recognizer import start_recognition
from flask import Flask, render_template_string, Response
import cv2
import push_notifications
import threading
import serial  # For Arduino communication
import time

# Arduino serial port (change COM3 to your port or /dev/ttyUSB0 on Linux)
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Allow Arduino to initialize

# Change to your dataset path
DATASET_PATH = r"D:\MECH\PYTHON\Engineering-teamwork---Rover\RasPi Codes\PHOTOS"
MODEL_PATH = "trained_model.yml"
CAMERA_INDEX = 0  # Change if needed

app = Flask(__name__)

# Start video capture (change index if needed)
cap = cv2.VideoCapture(2)

# HTML Template with Rover status
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Rover Control + Live Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body {
            background: #121212;
            color: #fff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 {
            font-size: 2em;
            margin-bottom: 10px;
        }
        .status {
            font-size: 1.5em;
            margin-bottom: 20px;
            color: #00ff88;
        }
        .video-container {
            border: 4px solid #444;
            border-radius: 12px;
            overflow: hidden;
            max-width: 90vw;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }
        img {
            width: 100%;
            height: auto;
            display: block;
        }
        .buttons {
            display: flex;
            gap: 20px;
            margin-top: 30px;
        }
        button {
            background: #1f1f1f;
            color: #fff;
            border: 2px solid #fff;
            border-radius: 8px;
            padding: 12px 24px;
            font-size: 1.1em;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            background: #fff;
            color: #121212;
            border-color: #ccc;
        }
    </style>
</head>
<body>
    <h1>📷 Live Rover Camera</h1>
    <div class="status">Status: {{ status }}</div>
    <div class="video-container">
        <img src="/video" alt="Rover Camera Stream" />
    </div>
    <div class="buttons">
        <form action="/start" method="post">
            <button type="submit">▶ Start Rover</button>
        </form>
        <form action="/stop" method="post">
            <button type="submit">■ Stop Rover</button>
        </form>
    </div>
</body>
</html>
"""

# Global variable for rover status
rover_status = "Checking..."

@app.route("/")
def index():
    return render_template_string(HTML, status=rover_status)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to grab frame")
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route("/video")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/start", methods=["POST"])
def start_rover():
    print("Rover started!")
    push_notifications.send_email("Rover Alert", "Package on the way!!")
    arduino.write(b'S')  # Tell Arduino to start line following
    return ""

@app.route("/stop", methods=["POST"])
def stop_rover():
    print("Rover stopped!")
    arduino.write(b'F')  # Tell Arduino to stop line following
    return ""

# Thread to continuously read from Arduino and update status
def read_arduino():
    global rover_status
    while True:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode().strip()
            if "ROVER LOADED" in line:
                rover_status = "LOADED"
            elif "ROVER EMPTY" in line:
                rover_status = "EMPTY"

# Start serial reading in background
threading.Thread(target=read_arduino, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
