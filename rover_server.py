from flask import Flask, render_template_string
import os

app = Flask(__name__)

# Simple HTML with buttons and video
HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Rover Control</title>
  <style>
    body { font-family: sans-serif; text-align: center; background: #f5f5f5; }
    button { padding: 12px 24px; margin: 10px; font-size: 16px; }
    #video { border: 2px solid #ccc; border-radius: 10px; width: 90%; max-width: 480px; }
  </style>
</head>
<body>
  <h1>📦 Rover Control Panel</h1>
  <img id="video" src="http://{{ ip }}:8080/?action=stream" alt="Webcam Feed" />
  <div>
    <form action="/start" method="post"><button type="submit" style="background:green;color:white;">Start</button></form>
    <form action="/stop" method="post"><button type="submit" style="background:red;color:white;">Stop</button></form>
  </div>
</body>
</html>
"""

@app.route("/")
def index():
    # Automatically uses the Pi's own IP
    return render_template_string(HTML_PAGE, ip=os.popen("hostname -I").read().split()[0])

@app.route("/start", methods=["POST"])
def start_rover():
    print("Rover started")
    # You can add GPIO control here
    return "Started. <a href='/'>Back</a>"

@app.route("/stop", methods=["POST"])
def stop_rover():
    print("Rover stopped")
    # You can add GPIO stop logic here
    return "Stopped. <a href='/'>Back</a>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
