# Face Recognition-1

import cv2

# Find cascades here (https://github.com/opencv/opencv/tree/master/data/haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Loading the cascade for the face.
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# Loading the cascade for the eyes.
smile_Cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
# Loading the cascade for the smile.

def detect(gray, frame): 
    # We create a function that takes as input, the image in black and white (gray), and the original image (frame)
    # and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(200, 200),flags=cv2.CASCADE_SCALE_IMAGE)
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    # 1.1 is the size of increasing the size of the filters/kernals. (Value got by experimenting)
    # 5 is the no of minimum neighbours, ie, the zone around the accepted zone.
    for (x, y, w, h) in faces:
        # x,y are co ordinates of upper left point.
        # w is the width of the selection of rectangle
        # h is the height of the selection of rectangle
        # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        # Frame, upper left point, lower right point, rgb code for a color, thickness of the edges of the rectangle
        # We paint a rectangle around the face.
        roi_gray = gray[y:y+h, x:x+w] 
        # roi is region of interest, ie, the region where the face is detected, ie we're going to find eyes only
        # in the region where we detected the face.
        # We get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] 
        # We get the region of interest in the colored image.
        cv2.putText(frame,'Face',(x, y), font, 2,(255,0,0),5)
        smile = smile_Cascade.detectMultiScale(roi_gray,scaleFactor= 1.75,minNeighbors=35,
                                              minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sh, sy), (sx+sw, sy+sh), (255, 0, 0), 2)
            cv2.putText(frame,'Smile',(x + sx,y + sy), 1, 1, (0, 255, 0), 1)
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: 
            # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
            cv2.putText(frame,'Eye',(x + ex,y + ey), 1, 1, (0, 255, 0), 1)
        # We paint a rectangle around the eyes, but inside the referential of the face.
        cv2.putText(frame,'Number of Faces : ' + str(len(faces)),(40, 40), font, 1,(255,0,0),2)  
    return frame 
# We return the image with the detector rectangles.
font = cv2.FONT_HERSHEY_SIMPLEX
video_capture = cv2.VideoCapture(0)
# 0 to turn on the inbuilt webcam, 1 to turn on external webcam connected. 
# We turn the inbuilt webcam on.

while True: 
    _, frame = video_capture.read() 
    # We get the last frame of the webcam.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # We convert the last frame to gray using the cv2.cvtcolor and cv2.COLOR_BGR2GRAY .
    canvas = detect(gray, frame) 
    # We get the output of our detect function.
    cv2.imshow('Video', canvas) 
    # We convert the frames(images) from detect function as a video.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # If we type 'q' on the keyboard:
        break 

video_capture.release() 
# We turn the webcam off.
cv2.destroyAllWindows() 
# We destroy all the windows inside which the images were displayed.
