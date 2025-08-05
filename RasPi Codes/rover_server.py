from flask import Flask, render_template_string, Response
import cv2
import push_notifications
import threading
import serial
import time
import pyttsx3

# === TTS Setup ===
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 135)
tts_engine.setProperty('volume', 1)

def speak(text):
    def run():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS error: {e}")
    threading.Thread(target=run).start()

# === Arduino Setup ===
arduino = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)

# === Camera Setup ===
cap = cv2.VideoCapture(1)

# === Global Rover Status ===
rover_status = "Checking..."

# === HTML Interface ===
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
            width: 800px;
            height: 600px;
            border: 4px solid #444;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 0 15px rgba(255, 255, 255, 0.1);
        }
        img {
            width: 100%;
            height: 100%;
            object-fit: cover;
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
            width: 220px;
            text-align: center;
        }
        button:hover {
            background: #fff;
            color: #121212;
            border-color: #ccc;
        }
    </style>
</head>
<body>
    <h1> Live Rover Camera</h1>
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
        <form action="/return" method="post">
            <button type="submit">↩ Return to Station</button>
        </form>
    </div>
</body>
</html>
"""

# === Flask App ===
app = Flask(__name__)

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
    speak("a rover started")
    arduino.write(b'S')
    push_notifications.send_email("Rover Alert", "Package on the way!!")
    return ""

@app.route("/stop", methods=["POST"])
def stop_rover():
    print("Rover stopped!")
    speak("a rover is stopping")
    arduino.write(b'F')
    return ""

@app.route("/return", methods=["POST"])
def return_rover():
    print("Rover returning to station...")
    speak("Rover returning to station")
    arduino.write(b'R')
    push_notifications.send_email("Rover Alert", "Rover returning to base.")
    return ""

# === Update Rover Status from Arduino Serial ===
def read_arduino():
    global rover_status
    while True:
        if arduino.in_waiting > 0:
            line = arduino.readline().decode().strip()
            print("Arduino:", line)
            if "ROVER LOADED" in line:
                rover_status = "LOADED"
            elif "ROVER EMPTY" in line:
                rover_status = "EMPTY"

# === Background Thread to Read Serial ===
threading.Thread(target=read_arduino, daemon=True).start()

# === Suppress Internal Error Page ===
@app.errorhandler(500)
def handle_500_error(e):
    return "", 204  # No redirect, no error page

# === Run App ===
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
