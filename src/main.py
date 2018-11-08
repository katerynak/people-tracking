#!/usr/bin/python3

import cv2

from bg_subtractor import Bg_subtractor

# background subtraction model
bg_sub = Bg_subtractor()

cap = cv2.VideoCapture('../pedestrians.mp4')
width = int(cap.get(3))
height = int(cap.get(4))

play, frame = cap.read()

while(cap.isOpened() and play):

    # extract the foreground mask
    mask = bg_sub.fg_mask(frame)

    # apply it and obtain an rgb foreground
    col_foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # display the image
    cv2.imshow('video', col_foreground)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    play, frame = cap.read()

cap.release()
cv2.destroyAllWindows()