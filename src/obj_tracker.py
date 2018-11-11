#!/usr/bin/python3

import cv2
import numpy as np
from random import randint
from scipy.spatial import distance_matrix, distance
from distances import distance_contours, arbitrary_distance_matrix

    #
    #
    # kf = cv2.KalmanFilter(4, 2)
    # kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    # kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    # def Estimate(self, coordX, coordY):
    #     """
    #     This function estimates the position of the object
    #     :param coordX:
    #     :param coordY:
    #     :return:
    #     """
    #     measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
    #     self.kf.correct(measured)
    #     predicted = self.kf.predict()
    #     return predicted

# def getKeys(mydict, value):
#     """
#     given a dictionary and a value returns a list of keys associated with that value
#     :param dict: dicitonary
#     :param value: value
#     :return:
#     """
#     ret = []
#
#     for key, val in mydict.items():
#         if val == value:
#             ret.append(key)
#
#     return ret


class Obj_tracker(object):
    def __init__(self, frameWidth, frameHeight):
        # param init
        # array of kalman filters as motion model
        # motion prediction and correction
        self.kfs = []
        # kf initialization parameters
        self.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

        # memory of ids from previous frames
        # list of dictionaries of type blobID:IDs
        self.lastIDs = []
        self.lastIDs.append({0: [0]})
        # list of last bboxes, each blobID corresponds to the index in the list
        self.lastBboxes = []
        # first bbox : whole frame
        self.lastBboxes.append([0, 0, frameWidth, frameHeight])
        # centers of bboxes
        self.lastBboxesCnt = []
        self.lastBboxesCnt.append(np.array([[frameWidth//2, frameHeight//2]]))
        # next available ID
        self.nextID = 1
        # list of dictionaries of type ID:list of counts
        self.colHists = []
        # number of frames to keep track of
        self.memSize = 2

        # distance threshold for id assignment, max distance a person can travel between 2 frames
        self.distThreshold = 50
        # trying out contours

        self.lastContours = []
        self.lastContours.append(np.array([[[[0,    0]], [[0,  719]], [[1279,  719]], [[1279,    0]]]]))

    def assignIDs(self, bboxes, frame):
        """
        assigns IDs to the bboxes
        :param bboxes:
        :param frame:
        :return:
        """
        bboxes = np.array(bboxes)

        # compute centers of bounding boxes : x + w/2 , y + h//2
        bboxes_cnt = [list(bboxes[:, 0] + bboxes[:,2]//2), list(bboxes[:, 1] + bboxes[:,3]//2)]
        # transpose list of lists to get [x,y] couples
        bboxes_cnt = list(map(list, zip(*bboxes_cnt)))

        # compute distances
        distances = arbitrary_distance_matrix(bboxes_cnt, self.lastBboxesCnt[-1], distance.euclidean)

        # assigning boxID to each ID
        # for each previous bbox assign it's id to the closest new box if
        # the distance between them exceeds a threshold
        dictids = dict()

        for i in range(len(bboxes)):
            dictids[i] = []

        dist_T= distances.transpose()
        # assign previous IDs to new boxes
        i=0
        for lastids, dist in zip(self.lastIDs[-1], dist_T):
            mindist = np.min(dist)
            if mindist <= self.distThreshold:
                # assign id of the closest next bbox
                for el in self.lastIDs[-1][i]:
                    if el not in dictids[np.argmin(dist)]:
                        dictids[np.argmin(dist)].append(el)
            i += 1

        i = 0
        for bbox, dist in zip(bboxes, distances):
            if len(dictids[i])==0:
                mindist = np.min(dist)
                # in case of splitting of a single blob assign to each blob all previous ids
                if mindist <= self.distThreshold:
                    for el in self.lastIDs[-1][np.argmin(dist)]:
                        if el not in dictids[i]:
                            dictids[i].append(el)
                else:
                    dictids[i].append(self.nextID)
                    self.nextID += 1
            i += 1

        # update memory
        self.lastIDs.append(dictids)
        self.lastBboxesCnt.append(bboxes_cnt)
        self.lastBboxes.append(bboxes)
        if len(self.lastBboxesCnt) > self.memSize:
            del(self.lastBboxesCnt[0])
            del(self.lastBboxes[0])
            del(self.lastIDs[0])

        return list(dictids.values())

    def assignIDsContours(self, contours, frame):
        """
        assigns IDs to the contours
        :param contours:
        :param frame:
        :return:
        """
        contours = np.array(contours)

        print(self.lastContours[-1])
        print(contours)

        # compute distances
        distances = arbitrary_distance_matrix(contours, self.lastContours[-1], distance_contours)

        # assigning boxID to each ID
        # for each previous bbox assign it's id to the closest new box if
        # the distance between them exceeds a threshold
        dictids = dict()

        for i in range(len(contours)):
            dictids[i] = []

        dist_T = distances.transpose()
        # assign previous IDs to new boxes
        i = 0
        for lastids, dist in zip(self.lastIDs[-1], dist_T):
            mindist = np.min(dist)
            if mindist <= self.distThreshold:
                # assign id of the closest next bbox
                for el in self.lastIDs[-1][i]:
                    if el not in dictids[np.argmin(dist)]:
                        dictids[np.argmin(dist)].append(el)
            i += 1

        i = 0
        for contour, dist in zip(contours, distances):
            if len(dictids[i]) == 0:
                mindist = np.min(dist)
                # in case of splitting of a single blob assign to each blob all previous ids
                if mindist <= self.distThreshold:
                    for el in self.lastIDs[-1][np.argmin(dist)]:
                        if el not in dictids[i]:
                            dictids[i].append(el)
                else:
                    dictids[i].append(self.nextID)
                    self.nextID += 1
            i += 1

        # update memory
        self.lastIDs.append(dictids)
        self.lastContours.append(contours)
        if len(self.lastIDs) > self.memSize:
            del (self.lastContours[0])
            del (self.lastIDs[0])

        return list(dictids.values())

