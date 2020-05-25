Hi All,

Created a basic smile detector using the Haar Cascades.

Find the cascades of OpenCV using https://github.com/opencv/opencv/tree/master/data/haarcascades 

Due to mustache, I've set the scale factor to 1.75 in the below line. try changing according to your need.

#smile = smile_Cascade.detectMultiScale(roi_gray,scaleFactor= 1.75,minNeighbors=35,minSize=(25, 25),flags=cv2.CASCADE_SCALE_IMAGE)

Happy coding.
