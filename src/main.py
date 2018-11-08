#!/usr/bin/python3

import cv2

from bg_subtractor import Bg_subtractor
from obj_detector import Obj_detector

# background subtraction model
bg_sub = Bg_subtractor()

# pedestrians detection model
ped_det = Obj_detector()

cap = cv2.VideoCapture('../pedestrians.mp4')
width = int(cap.get(3))
height = int(cap.get(4))

play, frame = cap.read()

while(cap.isOpened() and play):

    # extract the foreground mask
    mask = bg_sub.fg_mask(frame)

    # # apply it and obtain an rgb foreground
    # col_foreground = cv2.bitwise_and(frame, frame, mask=mask)

    # extract contours of pedestrians
    contours = ped_det.detect_objects(mask)

    cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

    # display the image
    cv2.imshow('video', frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    play, frame = cap.read()

cap.release()
cv2.destroyAllWindows()