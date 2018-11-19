#!/usr/bin/python3

import cv2
import numpy as np

from scipy.spatial import distance


class Obj_detector(object):
    def __init__(self):
        # parameter initialization
        self.bbox_area_min = 100

        # keeps in memory last contours detected
        self.contours = []
        self.bboxes = []

        self.sift = cv2.xfeatures2d.SIFT_create(nfeatures= 20, nOctaveLayers = 3, contrastThreshold = 0.01,
                                                edgeThreshold = 10, sigma = 1.6 )

    def select_contours(self, contours):
        """
        given a list of contours selects only those which are most significant in our application
        :param contours:
        :param threshold:
        :return:
        """
        selected = []
        for c in contours:
            bbox = cv2.boundingRect(c)
            x, y, w, h = bbox
            area = w * h
            if area > self.bbox_area_min:
                selected.append(c)
        return selected

    def approx_contours(self, contours):
        """
        returns an approximation of contours
        :param contours:
        :return:
        """
        approx = []
        for cnt in contours:
            epsilon = 0.1 * cv2.arcLength(cnt, True)
            approx.append(cv2.approxPolyDP(cnt, epsilon, True))
        return approx

    def detect_objects(self, mask):
        """
        detects object from a given binary mask, identify their contours
        :param mask:
        :return: list of contours of objects present in the binary mask
        """
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        self.contours = self.select_contours(contours)

        # contours approximation
        # self.contours = self.approx_contours(self.contours)

        return self.contours

    def get_bboxes(self):
        """
        returns rotated bounding boxes of last detected contours
        :return:
        """
        self.bboxes = []
        for cnt in self.contours:
            bbox = cv2.boundingRect(cnt)

            # rect = cv2.minAreaRect(cnt)
            # box = cv2.boxPoints(rect)
            # box = np.int0(box)
            self.bboxes.append(bbox)
        return self.bboxes

    def detect_keypoints(self, frame, draw=True):
        """
        given a grayscale frame detects keypoints
        :param frame: grayscale image
        :return: return
        """
        kp = self.sift.detect(frame, None)
        if draw:
            frame = cv2.drawKeypoints(frame, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return frame, kp


if __name__ == "__main__":

    obj_det = Obj_detector()

    from bg_subtractor import Bg_subtractor
    bg_sub = Bg_subtractor()

    cap = cv2.VideoCapture('../pedestrians.mp4')
    width = int(cap.get(3))
    height = int(cap.get(4))

    # create a named windows and move it
    cv2.namedWindow('video')
    cv2.moveWindow('video', 70, 30)

    next_frame, frame = cap.read()
    play = True

    while cap.isOpened() and next_frame:
        if play:

            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            # bg sub + contours
            mask = bg_sub.fg_mask(frame)
            contours = obj_det.detect_objects(mask)

            # draw contours
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

            # frame, kp = obj_det.detect_keypoints(frame)

            # bboxes extraction + draw
            bboxes = obj_det.get_bboxes()
            cv2.drawContours(frame, bboxes, -1, (0, 0, 255), 2)

            cv2.imshow('features', frame)
            next_frame, frame = cap.read()

        # q for exit, space for pause
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == 0x20:
            play = not play

    cap.release()
    cv2.destroyAllWindows()
