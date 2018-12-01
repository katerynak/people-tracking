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
import pandas as pd


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
rect = Rectifier(width, height, shift=50, up=True, central_zoom=80)

# create a named windows and move it
cv2.namedWindow('video')
cv2.moveWindow('video', 70, 30)

next_frame, frame = cap.read()
play = True

# pedestrians tracking model
ped_tr = Obj_tracker(width, height)


def drawBboxes(bboxes, frame, color = (0, 0, 255), ids=None):
    if ids is not None and len(ids)>0:
        for (bbox, id) in zip(bboxes, ids):
            x, y, w, h = bbox
            # draw a green rectangle to visualize the bounding rect
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, '{}'.format(id),
                        (x, y),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

    else:
        for bbox in bboxes:
            x, y, w, h = bbox
            # draw a green rectangle to visualize the bounding rect
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame

# pick random 1000 colors
colors = np.random.choice(range(256), size=(1000, 3))

ids = []

frameCnt = 1

out1 = "../selected_pedestrians_data/pedestrian10.csv"
out2 = "../selected_pedestrians_data/pedestrian36.csv"
out3 = "../selected_pedestrians_data/pedestrian42.csv"

data1 = []
data2 = []
data3 = []

id1 = 11
id2 = 171
id3 = 186

# out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 15, (width,height))

t0 = time.time()

while cap.isOpened() and next_frame:
    if play:

        # extract the foreground mask
        mask = bg_sub.fg_mask(frame)

        contours = ped_det.detect_objects(mask)
        bboxes = ped_det.get_bboxes()

        # extract foreground only for correct color histogram extraction
        h_foreground = cv2.bitwise_and(bg_sub.h, bg_sub.h, mask=mask)

        if len(bboxes) > 0:
            ids = ped_tr.assignIDs(bboxes, h_foreground)

        en, ex = stats.count_entering_exiting(ped_tr.bboxes_cnt, ids,
                                              ped_tr.idsVelocities, width, ped_tr.idsFirstFrame, ped_tr.frameCnt)



        if [id1] in ids:
            idx = ids.index([id1])
            data1.append({'frame': frameCnt, 'x': ped_tr.bboxes_cnt[idx][0], 'y': ped_tr.bboxes_cnt[idx][1]})
            # print(ped_tr.idsPositions[11][0])

        if [id2] in ids:
            idx = ids.index([id2])
            data3.append({'frame': frameCnt, 'x': ped_tr.bboxes_cnt[idx][0], 'y': ped_tr.bboxes_cnt[idx][1]})

        if [id3] in ids:
            idx = ids.index([id3])
            data2.append({'frame': frameCnt, 'x': ped_tr.bboxes_cnt[idx][0], 'y': ped_tr.bboxes_cnt[idx][1]})

        # update statistics
        stats.update(contours)

        # mark selected pedestrians

        # 10
        if frameCnt >= 0 and frameCnt < 5:
            frame = cv2.circle(frame, (730, 350), 15, (255, 0, 255), -1)

        # 36
        if frameCnt >= 264 and frameCnt < 271:
            frame = cv2.circle(frame, (1280, 280), 15, (255, 0, 255), -1)

        # 42
        if frameCnt >= 253 and frameCnt < 258:
            frame = cv2.circle(frame, (4, 581), 15, (255, 0, 255), -1)

        # draw contours
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        frame = drawBboxes(bboxes,frame, (0, 0, 255), ids=ids)

        for lastIDsPositons in ped_tr.lastIDsPositions:
            for id, pos in lastIDsPositons.items():
                posx, posy = pos
                r = int(colors[id][0])
                g = int(colors[id][1])
                b = int(colors[id][2])
                frame = cv2.circle(frame, (posx, posy), 5, (r, g, b), -1)

        # draw ped counts: detected / ground truth
        cv2.putText(frame, 'Objetcs: {}/{}'.format(len(contours), stats.get_curr_truth_counts()),
                    (int(width*0.30), int(height*0.12)),
                    cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)

        cv2.putText(frame, 'frame: {}'.format(frameCnt),
                    (int(width * 0.70), int(height * 0.12)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (80, 80, 80), 3)

        cv2.putText(frame, 'entering: {}'.format(en),
                    (int(width * 0.430), int(height * 0.90)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        cv2.putText(frame, 'exiting: {}'.format(ex),
                    (int(width * 0.430), int(height * 0.95)),
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 3)

        # drawing gates
        cv2.line(frame, (stats.gate_size, 0), (stats.gate_size, height), (88, 24, 69), 2)
        cv2.line(frame, (width - stats.gate_size, 0), (width - stats.gate_size, height), (88, 24, 69), 2)

        # display the image
        cv2.imshow('video', frame)

        # out.write(frame)

        next_frame, frame = cap.read()

        # time.sleep(0.1)
        frameCnt += 1

    # q for exit, space for pause
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == 0x20:
        play = not play

t1 = time.time()

total_time = t1-t0
print("Total time: {}".format(total_time))
print("Frames per second: {}".format(frameCnt/total_time))

stats.print_stats()

# pd.DataFrame(data1).to_csv(out1, header=None, index=False)
# pd.DataFrame(data2).to_csv(out2, header=None, index=False)
# pd.DataFrame(data3).to_csv(out3, header=None, index=False)


cap.release()
cv2.destroyAllWindows()