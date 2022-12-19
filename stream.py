# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:44:43 2022

@author: przem
"""

from flask import Flask, render_template, request, Response
import threading
import time
import cv2
import numpy as np

global vs, Frame, Mod_Frame, lock, faceDetection, setText, reducePixels, blurring

time.sleep(2.0)
vs = cv2.VideoCapture(1, cv2.CAP_DSHOW)
vs.set(cv2.CAP_PROP_POS_FRAMES, 0)
lock = threading.Lock()

Mod_Frame = None
faceDetection = False
setText = ''
showIdol = False
reducePixels = False
blurring = False

app = Flask(__name__, template_folder='')

@app.route("/", methods=['GET', 'POST'])
def index():
    global working, t, faceDetection, setText, showIdol, reducePixels, blurring
    if request.method == 'POST':
        if request.form.get('stopButton') == 'Stop':
            working=False
            t.join()
            shutdown_func = request.environ.get('werkzeug.server.shutdown')
            if shutdown_func is None:
                raise RuntimeError('Not running werkzeug')
            shutdown_func()
            
        if request.form.get('addTextButton') == 'Add text':
            setText = request.form['addTextInput']
            
        if request.form.get('reducePixelsButton') == 'Reduce pixels':
            if reducePixels == False:
                reducePixels = True
            else:
                reducePixels = False
                
        if request.form.get('showIdolButton') == 'Show your Idol':
            if showIdol == False:
                showIdol = True
            else:
                showIdol = False
                
        if request.form.get('faceDetectionButton') == 'Face detection':
            if faceDetection == False:
                faceDetection = True
            else:
                faceDetection = False
                
        if request.form.get('blurringButton') == 'Blurring':
            if blurring == False:
                blurring = True
            else:
                blurring = False
    
    elif request.method == 'GET':
        return render_template('index.html',form=request.form)    
    return render_template("index.html")

def generate():
    global Mod_Frame, lock
    while True:
        with lock:
            if Mod_Frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", Mod_Frame)
            if not flag:
                 continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def action():
    global setText, working, vs, Frame, Mod_Frame, lock
    
    maklowiczImage = cv2.imread('maklowicz.png')
    gS = cv2.cvtColor(maklowiczImage, cv2.COLOR_BGR2GRAY)
    
    while working:
        ret, frame = vs.read()
        ## tu dokonuj modyfikacji
        
        with lock:
            Mod_Frame = frame.copy()
            
            if reducePixels:
                height, width = Mod_Frame.shape[:2]
                toInterp = cv2.resize(Mod_Frame, (24, 24), interpolation=cv2.INTER_LINEAR)
                Mod_Frame = cv2.resize(toInterp, (width, height), interpolation=cv2.INTER_NEAREST)

            if blurring:
                Mod_Frame = cv2.blur(Mod_Frame, (20, 20))

            Mod_Frame = cv2.putText(Mod_Frame, setText, (160, 30), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (0, 255, 255))
            
            if showIdol:
                pi = Mod_Frame[-150 - 3:-3, -150 - 3:-3]
                pi[np.where(gS)] = 0
                pi += maklowiczImage
            
            if faceDetection:
                cc = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                toFaces = cv2.cvtColor(Mod_Frame, cv2.COLOR_BGR2GRAY)
                faces = cc.detectMultiScale(toFaces, 1.1, 9)
                for (x, y, w, h) in faces:
                    cv2.rectangle(Mod_Frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            

@app.route("/video_feed")
def video_feed():
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    global working,t
    working=True
    t = threading.Thread(target=action)
    t.daemon = True
    t.start()
    app.run(host="127.0.0.1", port="8080", debug=True,threaded=True, use_reloader=False)

    working=False 