#!/usr/bin/python3

import cv2

cap = cv2.VideoCapture('../pedestrians.mp4')
width = int(cap.get(3))
height = int(cap.get(4))

while(cap.isOpened()):
    play, frame = cap.read()
    cv2.imshow('video', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()