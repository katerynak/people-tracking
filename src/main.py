#!/usr/bin/python3

import cv2

from bg_subtractor import Bg_subtractor
from obj_detector import Obj_detector
from stats import Statistics
from rectifier import Rectifier

# background subtraction model
bg_sub = Bg_subtractor()

# pedestrians detection model
ped_det = Obj_detector()

# performance statistics
stats = Statistics("../groundtruth.txt")

cap = cv2.VideoCapture('../pedestrians.mp4')
width = int(cap.get(3))
height = int(cap.get(4))

# rectifying model
rect = Rectifier(width, height, 10)

# create a named windows and move it
cv2.namedWindow('video')
cv2.moveWindow('video', 70, 30)

next_frame, frame = cap.read()
play = True

while cap.isOpened() and next_frame:
    if play:

        frame = rect.rectify(frame)

        # extract the foreground mask
        mask = bg_sub.fg_mask(frame)

        # # apply it and obtain an rgb foreground
        # col_foreground = cv2.bitwise_and(frame, frame, mask=mask)

        # extract contours of pedestrians
        contours = ped_det.detect_objects(mask)

        # update statistics
        stats.update(contours)

        # draw contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

        # draw ped counts: detected / ground truth
        cv2.putText(frame, 'Objetcs: {}/{}'.format(len(contours), stats.get_curr_truth_counts()),
                    (int(width*0.30), int(height*0.12)),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)

        # display the image
        cv2.imshow('video', frame)

        next_frame, frame = cap.read()

    # q for exit, space for pause
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 0x20:
        play = not play


stats.print_stats()

cap.release()
cv2.destroyAllWindows()