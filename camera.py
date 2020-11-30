# import the necessary packages
import cv2
# defining face detector

try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
except Exception:
    print("Error loading cascade classifiers")

class VideoCamera(object):
    def __init__(self):
       #capturing video
       self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        #releasing camera
        self.video.release()

    def get_frame(self):
       # capture frome-by-frame:
        ret, frame = self.video.read()

        faces = face_cascade.detectMultiScale(image=frame, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            
            # The following are the parameters of cv2.rectangle()
            # cv2.rectangle(image_to_draw_on, start_point, end_point, color, line_width)
            cv2.rectangle(img=frame, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            
            roi = frame[y:y+h, x:x+w]
            
            # Detecting eyes in the face(s) detected
            eyes = eye_cascade.detectMultiScale(roi)
            
            # Detecting smiles in the face(s) detected
            smile = smile_cascade.detectMultiScale(roi, minNeighbors = 25)
            
            # Drawing rectangle around eyes
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi, (ex, ey), (ex+ew, ey+eh), (0,255,0), 2)
                
            # Drawing rectangle around smile
            for (sx,sy,sw,sh) in smile:
                cv2.rectangle(roi, (sx, sy), (sx+sw, sy+sh), (0,0,255), 2)
            break

        # display frame
        # cv2.imshow('Video', frame)
    
        # encode OpenCV raw frame to jpg and displaying it
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()