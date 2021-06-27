# import the necessary packages
from pyimagesearch.motion_detection.singlemotiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

outputFrame = None
lock = threading.Lock()
app = Flask(__name__)
vs = VideoStream(src=0).start()
time.sleep(2.0)


@app.route("/")
def index():
    return render_template("index.html")


def detect_motion(frameCount, option):
    global vs, outputFrame, lock
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        cv2.putText(img=frame, text="jaroslaw kudzia", org=(100, 250), fontFace=cv2.FONT_ITALIC,
                    fontScale=1, color=(245, 108, 66), thickness=2)
        if(option == 'motion'):
            if total > frameCount:
                motion = md.detect(gray)
                if motion is not None:
                    (thresh, (minX, minY, maxX, maxY)) = motion
                    cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                                  (0, 0, 255), 2)
        elif(option == 'szarosc'):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif(option == 'blur'):
            frame = cv2.blur(frame, (10,30))
        elif (option == 'edge'):
            frame = cv2.Canny(frame, 50, 100)

        md.update(gray)
        total += 1
        with lock:
            outputFrame = frame.copy()


def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--ip", type=str, required=True,
                help="ip address of the device")
ap.add_argument("-o", "--port", type=int, required=True,
                help="ephemeral port number of the server (1024 to 65535)")
ap.add_argument("-f", "--frame-count", type=int, default=32,
                help="# of frames used to construct the background model")
args = vars(ap.parse_args())

########## JAKI TRYB STEAMINGu
# szarosc, motion, blur, edge
option = 'blur'



# start a thread that will perform motion detection
t = threading.Thread(target=detect_motion, args=(
    args["frame_count"],option))
t.daemon = True
t.start()
# start the flask app
app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
# release the video stream pointer
vs.stop()
