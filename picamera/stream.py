# ============================================================
# qr_stream.py
# Live QR detection stream from the Pi camera to a web browser.
#
# What it does:
#   1. Grabs frames from the Pi camera (Picamera2)
#   2. Runs OpenCV's QR detector on each frame
#   3. Draws the result onto the frame (box + status text)
#   4. Serves the annotated frames as a live video in a browser (Flask)
#
# Run on the Pi:      python3 qr_stream.py
# Watch on laptop:    http://192.168.2.40:5000   (in a web browser)
# Stop:               Ctrl+C in the Pi terminal
# ============================================================

import math                        # for the pixel-width calculation (hypot)
from flask import Flask, Response  # tiny web server + a way to send a stream
from picamera2 import Picamera2    # grabs frames from the Pi camera
import cv2                         # OpenCV: QR detection, drawing, JPEG encoding


# ---------- SETTINGS (edit these by hand) ----------
FRAME_WIDTH  = 1280    # set to the resolution you plan to FLY with
FRAME_HEIGHT = 720
PORT         = 5000    
JPEG_QUALITY = 70      # 0-100; lower = smaller frames = less lag over the network


# ---------- CAMERA SETUP (runs ONCE, before any loop) ----------
picam2 = Picamera2()                              # create the camera object
config = picam2.create_preview_configuration(
    main={"size": (FRAME_WIDTH, FRAME_HEIGHT),
          "format": "RGB888"})                    # 3-channel image, already in the
                                                  # color order OpenCV expects (BGR)
picam2.configure(config)                          # apply the settings
picam2.start()                                    # camera starts producing frames


# ---------- DETECTOR SETUP (also ONCE) ----------
detector = cv2.QRCodeDetector()                   # reusable detector object


# ---------- WEB SERVER SETUP ----------
app = Flask(__name__)                             # create the web server object


def annotate(frame):
    """Run QR detection on one frame and draw the result onto it.
    Input:  frame (image from the camera)
    Output: the same frame with a box + status text drawn on it
    """
    # Detector outputs (we only give it the image):
    #   data -> decoded text, "" if decoding failed
    #   bbox -> 4 corner points, shape (1, 4, 2), or None if nothing found
    #   _    -> straightened QR image (not needed, so ignored)
    data, bbox, _ = detector.detectAndDecode(frame)

    # --- Sort into the three outcomes worth telling apart ---
    if bbox is None:
        status = "NOT FOUND"
        color = (0, 0, 255)                       # red   (colors are B, G, R)
    elif data == "":
        status = "FOUND, NOT DECODED"             # the interesting edge-of-range case
        color = (0, 255, 255)                     # yellow
    else:
        status = "DECODED: " + data
        color = (0, 255, 0)                       # green

    # --- If we have corners, draw the box and measure the width ---
    if bbox is not None:
        corners = bbox[0].astype(int)             # 4 corners as whole-number pixels

        # draw the 4 sides: corner i connects to corner i+1 (wrapping 3 -> 0)
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, p1, p2, color, 3)

        # pixel width = distance between two neighboring corners (0 and 1)
        dx = corners[1][0] - corners[0][0]
        dy = corners[1][1] - corners[0][1]
        width_px = math.hypot(dx, dy)             # sqrt(dx^2 + dy^2)

        cv2.putText(frame, f"width: {width_px:.0f} px", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # status text in the top-left corner
    cv2.putText(frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    return frame


def generate_frames():
    """Endless loop: grab -> detect/annotate -> compress -> send.
    'yield' hands one frame to the browser, then the loop keeps going.
    """
    while True:
        frame = picam2.capture_array()            # newest camera frame
        frame = annotate(frame)                   # detection + drawing

        # compress to JPEG so it's small enough to send quickly
        ok, buffer = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue                              # skip a frame that failed to encode

        # each frame is sent as one "part" of a never-ending multipart response;
        # the browser replaces the old image with each new one = live video
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ---------- WEB PAGES ----------
@app.route("/")
def index():
    # simple page that just displays the stream
    return '<html><body style="margin:0;background:#000">' \
           '<img src="/video_feed" style="width:100%"></body></html>'


@app.route("/video_feed")
def video_feed():
    # hand the browser the endless stream from generate_frames()
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------- START ----------
if __name__ == "__main__":
    try:
        # 0.0.0.0 = listen on all of the Pi's network connections
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        # runs even if you stop with Ctrl+C, so the camera is released cleanly
        picam2.stop()