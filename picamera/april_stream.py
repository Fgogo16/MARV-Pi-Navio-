# ============================================================
# apriltag_stream.py
# Live AprilTag detection stream from the Pi camera to a web browser.
#
# What it does:
#   1. Grabs frames from the Pi camera (Picamera2)
#   2. Runs the AprilTag detector (pupil-apriltags library) on each frame
#   3. Draws the result onto the frame (box, tag ID, width, confidence)
#   4. Serves the annotated frames as a live video in a browser (Flask)
#
# One-time install on the Pi (needs internet):
#   pip install pupil-apriltags --break-system-packages
#
# Run on the Pi:      python3 apriltag_stream.py
# Watch on laptop:    http://192.168.137.140:5000   (check the IP with: hostname -I)
# Stop:               Ctrl+C in the Pi terminal
# Only one program can use the camera at a time.
# ============================================================

import math                           # for the pixel-width calculation (hypot)
from flask import Flask, Response     # tiny web server + a way to send a stream
from picamera2 import Picamera2       # grabs frames from the Pi camera
import cv2                            # OpenCV: grayscale, drawing, JPEG encoding
from pupil_apriltags import Detector  # the AprilTag detector


# ---------- CAMERA SETTINGS (edit these by hand) ----------
FRAME_WIDTH  = 1536    # native fast mode for Camera Module 3; try 2304 x 1296 too
FRAME_HEIGHT = 864

# ---------- TAG / DETECTOR SETTINGS ----------
TAG_FAMILY   = "tag25h9"  # must match the family printed on your tags
QUAD_DECIMATE = 1.0        # 1.0 = search the full-resolution image (best range)
                           # 2.0 = search a half-size copy (faster, shorter range)
NUM_THREADS  = 4           # the Pi has 4 cores; let the detector use them

# ---------- NETWORK / STREAM SETTINGS ----------
PORT         = 5000    # the "apartment number" the browser connects to
JPEG_QUALITY = 70      # 0-100; only affects the browser view, NOT detection


# ---------- CAMERA SETUP (runs ONCE, before any loop) ----------
picam2 = Picamera2()
config = picam2.create_preview_configuration()                
                                                  # byte order OpenCV expects (BGR)
picam2.configure(config)
picam2.start()


# ---------- DETECTOR SETUP (also ONCE) ----------
detector = Detector(families=TAG_FAMILY,
                    quad_decimate=QUAD_DECIMATE,
                    nthreads=NUM_THREADS)


# ---------- WEB SERVER SETUP ----------
app = Flask(__name__)


def annotate(frame):
    """Run AprilTag detection on one frame and draw the result onto it.
    Input:  frame (image from the camera)
    Output: the same frame with boxes, IDs, and status text drawn on it
    """
    # The detector only works on black-and-white (grayscale) images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Returns a list with one entry per decoded tag (empty list if none).
    # Each entry has, among others:
    #   .tag_id          -> the tag's ID number
    #   .corners         -> 4 corner points in pixels, shape (4, 2)
    #   .decision_margin -> decode confidence; higher = cleaner read
    #   .hamming         -> bits the detector had to correct (0 is ideal)
    tags = detector.detect(gray)

    if len(tags) > 0:
        status = "DETECTED: " + ", ".join(str(t.tag_id) for t in tags)
        color = (0, 255, 0)                       # green   (colors are B, G, R)
    else:
        status = "NOT FOUND"
        color = (0, 0, 255)                       # red

    # --- For each tag: draw box, label ID, measure width, show confidence ---
    for t in tags:
        pts = t.corners.astype(int)               # 4 corners as whole-number pixels

        # draw the 4 sides; % 4 wraps the last corner back to the first
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), color, 3)

        # pixel width = distance between neighboring corners 0 and 1
        dx = pts[1][0] - pts[0][0]
        dy = pts[1][1] - pts[0][1]
        width_px = math.hypot(dx, dy)

        # label at the tag's center
        cx, cy = t.center.astype(int)
        label = f"ID {t.tag_id} | {width_px:.0f} px | margin {t.decision_margin:.0f}"
        cv2.putText(frame, label, (cx - 150, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # overall status in the top-left corner
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

        ok, buffer = cv2.imencode(".jpg", frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            continue                              # skip a frame that failed to encode

        # packaging so the browser plays the stream of JPEGs as video
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")


# ---------- WEB PAGES ----------
@app.route("/")
def index():
    # page that displays the stream, fit to the window width
    return '<html><body style="margin:0;background:#000">' \
           '<img src="/video_feed" style="width:100%"></body></html>'


@app.route("/video_feed")
def video_feed():
    # the raw endless stream from generate_frames()
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------- START ----------
if __name__ == "__main__":
    try:
        # 0.0.0.0 = listen on all of the Pi's network connections
        app.run(host="0.0.0.0", port=PORT, threaded=True)
    finally:
        # runs even on Ctrl+C, so the camera is always released
        picam2.stop()