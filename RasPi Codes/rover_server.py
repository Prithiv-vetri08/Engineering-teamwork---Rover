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
arduino = serial.Serial('COM6', 9600, timeout=1)
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
        #notification {
            margin-top: 20px;
            font-size: 1.2em;
            color: #00ff88;
        }
    </style>
</head>
<body>
    <h1>Live Rover Camera</h1>
    <div class="status">Status: {{ status }}</div>
    <div class="video-container">
        <img src="/video" alt="Rover Camera Stream" />
    </div>
    <div class="buttons">
        <button onclick="sendCommand('/start')">▶ Start Rover</button>
        <button onclick="sendCommand('/stop')">■ Stop Rover</button>
        <button onclick="sendCommand('/return')">↩ Return to Station</button>
    </div>
    <div id="notification"></div>

    <script>
        function sendCommand(endpoint) {
            fetch(endpoint, {
                method: 'POST'
            }).then(response => {
                if (response.ok) {
                    let message = "";
                    if (endpoint === "/start") {
                        message = "✅ Rover Started!";
                    } else if (endpoint === "/stop") {
                        message = "🛑 Rover Stopped.";
                    } else if (endpoint === "/return") {
                        message = "↩ Rover Returning to Station...";
                    }
                    showNotification(message);
                } else {
                    showNotification("⚠️ Command failed.", true);
                }
            }).catch(err => {
                console.error("Fetch error:", err);
                showNotification("❌ Error sending command.", true);
            });
        }

        function showNotification(message, isError = false) {
            const notify = document.getElementById("notification");
            notify.textContent = message;
            notify.style.color = isError ? "#ff4c4c" : "#00ff88";
            setTimeout(() => {
                notify.textContent = "";
            }, 4000);
        }

        function updateStatus() {
            fetch('/status')
                .then(response => response.text())
                .then(status => {
                    document.querySelector('.status').textContent = 'Status: ' + status;
                })
                .catch(error => console.error('Status update failed:', error));
        }

        setInterval(updateStatus, 1000); // update every 1 second
        updateStatus(); // initial call
    </script>
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
    speak("Rover started")
    arduino.write(b'S')
    push_notifications.send_email("Rover Alert", "Package on the way!!")
    return ""

@app.route("/stop", methods=["POST"])
def stop_rover():
    print("Rover stopped!")
    speak("Rover is stopping")
    arduino.write(b'F')
    return ""

@app.route("/return", methods=["POST"])
def return_rover():
    print("Rover returning to station...")
    speak("Rover returning to station")
    arduino.write(b'R')
    push_notifications.send_email("Rover Alert", "Rover returning to base.")
    return ""

@app.route("/status")
def get_status():
    return rover_status

# === Serial Listener Thread ===
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

threading.Thread(target=read_arduino, daemon=True).start()

@app.errorhandler(500)
def handle_500_error(e):
    return "", 204  # No redirect, no error page

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
