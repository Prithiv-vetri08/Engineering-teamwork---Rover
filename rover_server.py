from flask import Flask, render_template_string, Response
import cv2

app = Flask(__name__)

# Start video capture (0 = default USB webcam)
cap = cv2.VideoCapture(0)

# If you're using the Pi Camera Module (with libcamera), use:
# cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Rover Control + Live Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="text-align:center; font-family:sans-serif; background:#f0f0f0;">
    <h1>📷 Live Rover Camera</h1>
    <img src="/video" style="max-width:100%; height:auto; display:block; margin:auto; border:2px solid #ccc;" />
    <form action="/start" method="post">
        <button type="submit" style="padding:10px 20px; margin:10px; font-size:16px;">▶ Start</button>
    </form>
    <form action="/stop" method="post">
        <button type="submit" style="padding:10px 20px; margin:10px; font-size:16px;">■ Stop</button>
    </form>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

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
    print("Rover started!")  # Add GPIO logic here
    return "<h3>Rover Started</h3><a href='/'>Back</a>"

@app.route("/stop", methods=["POST"])
def stop_rover():
    print("Rover stopped!")  # Add GPIO logic here
    return "<h3>Rover Stopped</h3><a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
