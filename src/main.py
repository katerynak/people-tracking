#!/usr/bin/python3

import cv2
import numpy as np
import time

from bg_subtractor import Bg_subtractor
from obj_detector import Obj_detector
from obj_tracker import Obj_tracker
# from sort import Sort
from stats import Statistics
from rectifier import Rectifier


# background subtraction model
bg_sub = Bg_subtractor()

# pedestrians detection model
ped_det = Obj_detector()

# ped_tr = Sort()

# performance statistics
stats = Statistics("../groundtruth.txt")

cap = cv2.VideoCapture('../pedestrians.mp4')
width = int(cap.get(3))
height = int(cap.get(4))

# rectifying model
rect = Rectifier(width, height, shift=35, up=True, central_zoom=80)

# create a named windows and move it
cv2.namedWindow('video')
cv2.moveWindow('video', 70, 30)

next_frame, frame = cap.read()
play = True

# pedestrians tracking model
ped_tr = Obj_tracker(width, height)

def bbox2coords(bboxes):
    coords = np.zeros([len(bboxes), 4])
    for i, bbox in enumerate(bboxes):
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h
        coord = np.array([x1,y1,x2,y2])
        coords[i]=coord
    return coords


def coords2bbox(coords):
    bboxes = []
    ids=[]
    for coord in coords:
        x1, y1, x2, y2, id = coord
        ids.append(id)
        bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
        bboxes.append(bbox)
    return bboxes, ids


def drawBboxes(bboxes, frame, color = (0, 0, 255), ids=None):
    if ids is not None and len(ids)>0:
        for (bbox, id) in zip(bboxes, ids):
            x, y, w, h = bbox
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, '{}'.format(id),
                        (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    else:
        for bbox in bboxes:
            x, y, w, h = bbox
            # draw a green rectangle to visualize the bounding rect
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame

#
# cnt = 0
# cnt_upd= 2
# coords = np.array([[0.,   0., 1280.,  720.]])

ids = []

while cap.isOpened() and next_frame:
    if play:

        frame = rect.rectify(frame)

        # extract the foreground mask
        mask = bg_sub.fg_mask(frame)

        # mask = rect.rectify(mask)

        # frame = rect.rectify(frame)


        # apply it and obtain an rgb foreground
        # col_foreground = cv2.bitwise_and(frame, frame, mask=mask)

        contours = ped_det.detect_objects(mask)
        bboxes = ped_det.get_bboxes()

        h_foreground = cv2.bitwise_and(bg_sub.h, bg_sub.h, mask=mask)

        if len(bboxes) > 0:
            # ids = ped_tr.assignIDs(bboxes, bg_sub.h)
            ids = ped_tr.assignIDs(bboxes, h_foreground)
            # ids = ped_tr.assignIDs(bboxes, frame[2])
            # ids = ped_tr.assignIDsContours(contours, frame)

        # print(bboxes[0])
        # if cnt%cnt_upd==0 or cnt<10:
        #     # extract contours of pedestrians
        #     coords = bbox2coords(bboxes)

        # if len(coords)==0:
        #     coords = old_coords
        #
        # print("coords")
        # print(coords)

        # coords = bbox2coords(bboxes)
        # np.random.shuffle(coords)
        # coords = ped_tr.update(coords[:,:4].astype(int))
        # print(coords)
        #
        # if len(coords)>0:
        #     old_coords = coords
        #
        # cnt +=1
        #
        # predicted_bboxes, ids = coords2bbox(coords)

        # update statistics
        stats.update(contours)

        # draw contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        frame = drawBboxes(bboxes,frame, (0, 0, 255), ids=ids)

        # frame = drawBboxes(predicted_bboxes, frame, (255,0,0), ids=ids)
        # if len(bboxes)>0:
        #      cv2.drawContours(frame, bboxes, -1, (0, 0, 255), 2)

        # draw ped counts: detected / ground truth
        cv2.putText(frame, 'Objetcs: {}/{}'.format(len(contours), stats.get_curr_truth_counts()),
                    (int(width*0.30), int(height*0.12)),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)

        # display the image
        cv2.imshow('video', frame)
        # cv2.imshow('v', h_foreground)

        next_frame, frame = cap.read()
        time.sleep(0.1)

    # q for exit, space for pause
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 0x20:
        play = not play


stats.print_stats()

cap.release()
cv2.destroyAllWindows()