# Face Recognition

import cv2

# Find cascades here (https://github.com/opencv/opencv/tree/master/data/haarcascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# We load the cascade for the face.
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
# We load the cascade for the eyes.

#face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 
# We load the cascade for the face.
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml') 
# We load the cascade for the eyes.

def detect(gray, frame): 
    # We create a function that takes as input the image in black and white (gray) and the original image (frame), 
    # and that will return the same image with the detector rectangles. 
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) 
    # We apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    # 1.3 is the size of increasing the size of the filters/kernals. (Value got by experimenting)
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
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        # We apply the detectMultiScale method to locate one or several eyes in the image.
        for (ex, ey, ew, eh) in eyes: 
            # For each detected eye:
            cv2.rectangle(roi_color,(ex, ey),(ex+ew, ey+eh), (0, 255, 0), 2)
            # We paint a rectangle around the eyes, but inside the referential of the face.
    return frame 
# We return the image with the detector rectangles.

video_capture = cv2.VideoCapture(0)
# 0 to turn on the inbuilt webcam, 1 to turn on external webcam connected. 
# We turn the webcam on.

while True: 
    # We repeat infinitely (until break):
    _, frame = video_capture.read() 
    # We get the last frame of the webcam.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # We convert the last frame to gray using the cv2.cvtcolor and cv2.COLOR_BGR2GRAY .
    canvas = detect(gray, frame) 
    # We get the output of our detect function.
    cv2.imshow('Video', canvas) 
    # We convert the frames(images) from detect function as a video.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # If we type on the keyboard:
        break 
    # We stop the loop.

video_capture.release() 
# We turn the webcam off.
cv2.destroyAllWindows() 
# We destroy all the windows inside which the images were displayed.