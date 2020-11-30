# import the necessary packages
from flask import Flask, render_template, Response
import cv2
from camera import VideoCamera
app = Flask(__name__)

@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def gen(camera):
    while True:
        #get camera frame
        frame = camera.get_frame()
        k = cv2.waitKey(10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break   

        #ESC Pressed
        if k%256 == 27: 
            break
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    app.run(port=5000, debug=True)
    