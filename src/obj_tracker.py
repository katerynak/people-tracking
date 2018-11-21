#!/usr/bin/python3

import cv2
import numpy as np
from scipy.spatial import distance_matrix, distance
from distances import distance_contours, arbitrary_distance_matrix
from kalman_filter import KalmanFilter
import heapq
from operator import itemgetter
from sklearn.utils.linear_assignment_ import linear_assignment

# function to flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


def getKeys(mydict, value):
    """
    given a dictionary value: list and a value, returns a list of keys associated lists containing
    that value
    :param dict: dicitonary
    :param value: value
    :return:
    """
    ret = []

    for key, val in mydict.items():
        if value in val:
            ret.append(key)

    return ret


def getKeysWholeList(mydict, value):
    """
    given a dictionary value: list and a value, returns a list of keys associated lists containing
    that value
    :param dict: dicitonary
    :param value: value
    :return:
    """
    ret = []

    for key, val in mydict.items():
        if value == val:
            ret.append(key)

    return ret

def normalized_dot_product(v1, v2):
    """
    computes normalized dot product of two vectors
    :param v1:
    :param v2:
    :return:
    """

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    return np.dot(v1, v2)


class Obj_tracker(object):
    def __init__(self, frameWidth, frameHeight):
        # param init
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight


        # memory of ids from previous frames
        # list of dictionaries of type blobID:IDs
        self.lastIDs = []
        self.lastIDs.append({})
        self.lastIDs.append({})

        # if bounding boxes are used
        # list of last bboxes, each blobID corresponds to the index in the list
        self.lastBboxes = []
        # centers of bboxes
        self.lastBboxesCnt = []

        # if contours are used
        self.lastContours = []

        # next available ID
        self.nextID = 1

        # number of bins for color histograms
        self.bins = 64

        # list of histograms of single ids, id: hist
        self.idsHists = []
        self.idsHists.append({})

        # histograms of current bboxes
        self.hists = {}

        # positions of bounded boxes of current frame
        self.bboxes_cnt = {}

        # list of kalman filters of single ids
        # id: KalmanFilter instance
        self.kalmanFilters = {}

        # id: pos
        self.idsPositions = {}

        # id: lastFrame
        self.idsLastUpdate = {}
        self.idsFirstFrame = {}
        # counter for subsequent frames, for identifying temporary and well established ids
        # id: subsequent frames counter
        self.idsFramesCnt = {}
        self.establishedThresh = 5

        # list of established ids
        self.establishedIds = []
        # self.idsLastUpdate[0] = 0

        self.maxIdAge = 15

        # number of frames to keep track of
        self.memSize = 15

        # distance threshold for id assignment, max distance a person can travel between 2 frames
        self.distThreshold1 = 80
        # self.distThreshold = 2
        # trying out contours

        self.bbox_area_min_4 = 10000

        self.frameCnt = 0



    def histSimilarity(self, hist1, hist2, method=cv2.HISTCMP_INTERSECT, normalize=True):
        """
        compute color similarity between hist1 and hist2
        other methods are described here:
        https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist
        with the following notation for opencv3:
        cv2.HISTCMP_CORREL
        cv2.HISTCMP_CHISQR
        cv2.HISTCMP_INTERSECT
        cv2.HISTCMP_BHATTACHARYYA
        :param hist1:
        :param hist2:
        :return:
        """
        # hist normalization
        if normalize:
            h1 = hist1 / np.sum(hist1)
            h2 = hist2 / np.sum(hist2)

        return cv2.compareHist(h1, h2, method=method)

    def similarity(self, id, boxid, dist_weight = 0.5, velocity_weight = 0):
        """
        returns similarity of the box comparing to the statistics of the box corresponding to the target id
        in this function 2 similarity metrics are combined:
        1. color histogram similarity
        2. inverse distance similarity: 1/(predicted id position - box position)
        :param id:
        :param boxid:
        :return:
        """
        h1 = self.hists[boxid]
        if len(self.idsHists[-1].keys()) == 0:
            col_sim = 0
        else:
            h2 = self.idsHists[-1][id]
            col_sim = self.histSimilarity(h1, h2)
        box_x, box_y = self.bboxes_cnt[boxid]

        pos = self.kalmanFilters[id].lastPosition
        xp, yp = pos
        dist = distance.euclidean([box_x, box_y], [xp, yp])
        dist_sim = 1 / max(dist, 0.00001)
        return col_sim*(1-dist_weight) + dist_sim*dist_weight

    def most_similar(self, id, boxesIds, col_threshold = 0.1):
        """
        given an id and ids of the closest boxes
        returns the most similar to the pedestrian (according to color + euclidean distances) box id
        :param id:
        :param boxesIds:
        :return:
        """
        dists = np.zeros(len(boxesIds))
        i = 0
        for boxId in boxesIds:
            dists[i] = self.similarity(id, boxId, dist_weight=0.5)
            i += 1
        if np.max(dists>col_threshold):
            max_sim = boxesIds[np.argmax(dists)]
        else: max_sim = -1
        return max_sim

    def __update_positions(self, bboxes_cnt):

        for id in self.kalmanFilters:
            self.idsPositions[id] = self.kalmanFilters[id].predict()

    def __find_oldest_id(self, ids, n = 1):
        firstFrames={}
        for id in ids:
            firstFrames[id] = self.idsFirstFrame[id]

        return heapq.nsmallest(n, firstFrames, key=firstFrames.get)
        # return min(firstFrames, key=firstFrames.get)

    def assignIDs(self, bboxes, frame_h):
        """
        assigns IDs to the bboxes
        :param bboxes:
        :param frame_h: hue component of the frame
        :return:
        """
        self.frameCnt += 1

        bboxes = np.array(bboxes)

        # --------------- compute bbox coordinates --------------------------
        # compute centers of bounding boxes : x + w/2 , y + h//2
        bboxes_cnt = [list(bboxes[:, 0] + bboxes[:, 2] // 2), list(bboxes[:, 1] + bboxes[:, 3] // 2)]
        # transpose list of lists to get [x,y] couples
        bboxes_cnt = list(map(list, zip(*bboxes_cnt)))

        self.bboxes_cnt = bboxes_cnt

        # ---------------- updating kalman predictions ----------------------
        # update id positions based on Kalman filter predictions
        self.__update_positions(bboxes_cnt)


        # ---------------- computing current color histograms of all bboxes -
        # compute color histograms of the current boxes
        hists = []
        for bbox in bboxes:
            subframe = frame_h[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            # from 1 to 255 : we don't count the background mask info
            hist_h = cv2.calcHist([subframe], [0], None, [self.bins], [1, 230])
            hists.append(hist_h)

        self.hists = hists

        # ------------------- assign IDs to new boxes ----------------------------
        distances = arbitrary_distance_matrix(list(self.idsPositions.values()), bboxes_cnt, distance.euclidean)

        dictids = dict()

        for i in range(len(bboxes)):
            dictids[i] = []

        # bbox pool, mask used to assign available ids
        bbox_available_mask = np.array([True] * len(bboxes))

        for id, dist in zip(self.idsPositions, distances):
            dists = dist[bbox_available_mask]
            if len(dists) == 0:
                break
            idx = np.atleast_1d(np.argwhere(bbox_available_mask).squeeze())

            mindist = np.min(dists)
            if mindist <= self.distThreshold1:
                # assign id of the closest next bbox

                # select the closest 3 bboxes and assign an id to the most similar one
                orderedIdx = idx[np.argsort(dists)]
                ordered = np.sort(dists)

                closestBboxes = orderedIdx[ordered<self.distThreshold1][:3]

                most_sim = self.most_similar(id, closestBboxes)
                # most_sim = np.argsort(np.logical_not(bbox_available_mask))[most_sim]
                if most_sim != -1:
                    dictids[most_sim].append(id)
                    # dictids[np.argmin(dist)].append(id)
                    self.idsLastUpdate[id] = self.frameCnt
                    # bbox_available_mask[most_sim] = False

        # ------------------ assignment of new ids to boxes without ids -----------

        # check if some bboxes have lost all ids
        for boxid, ids in dictids.items():
            if len(ids)==0:
                dictids[boxid].append(self.nextID)
                x, y = bboxes_cnt[boxid]
                self.kalmanFilters[self.nextID] = KalmanFilter(x, y)
                self.idsLastUpdate[self.nextID] = self.frameCnt
                self.idsFirstFrame[self.nextID] = self.frameCnt
                self.nextID += 1

        # ------------------ a box can containg max 4 persons, if it contains more, cut out the youngest-
        # if a box contains 4 persons or more check also it's dimension, if it's too small cut the number of ids

        for boxid, ids in dictids.items():
            if len(ids) > 4:
                dictids[boxid] = self.__find_oldest_id(ids, n=4)
                idxToDel = list(set(ids) - set(dictids[boxid]))
                for idx in idxToDel:
                    del (self.idsPositions[idx])
                    del (self.idsLastUpdate[idx])
                    del (self.kalmanFilters[idx])

        for boxid, ids in dictids.items():
            if len(ids) == 4:
                print("----------------------")
                print(ids)
                print(bboxes[boxid][2] * bboxes[boxid][3])
                if bboxes[boxid][2] * bboxes[boxid][3] < self.bbox_area_min_4:
                    print(ids)
                    dictids[boxid] = self.__find_oldest_id(ids, n=2)
                    idxToDel = list(set(ids) - set(dictids[boxid]))
                    for idx in idxToDel:
                        del (self.idsPositions[idx])
                        del (self.idsLastUpdate[idx])
                        del (self.kalmanFilters[idx])

        # ------------------ updating histograms only of bboxes with single id ------------------
        # decide which color histograms to update: only an id assigned to a single bbox
        # and in case when it is the only id of that bbox
        # updating histograms, contains couples id:boxid
        idsToUpdate = {}
        for boxid, ids in dictids.items():
            # if box contains one single id
            if len(ids) == 1:
                # check if that id is contained only in one box
                if len(getKeys(dictids, ids[0])) == 1:
                    idsToUpdate[ids[0]] = boxid

        idsHists = self.idsHists[-1]
        # update the histograms of selected ids
        for id, boxid in idsToUpdate.items():
            cnt = 0
            # if id in idsHists.keys():
            #     # average with the previous value
            #     idsHists[id] *= 0.1
            #     idsHists[id] += hists[boxid]*0.9
            # else:
            idsHists[id] = hists[boxid]
            for phists in self.idsHists[::-1]:
                    for idx in phists:
                        if idx == id:
                            idsHists[id] += phists[id]
                            cnt += 1
            idsHists[id] /= cnt+1

        # ------------------- correcting Kalman filters: for box with single id, uses its position, for those with multiple ids uses
        #                     a combination between predicted position and weighted box position

        for id in list(set(flatten(dictids.values()))):
            x, y = bboxes_cnt[getKeys(dictids, id)[0]]
            # correct position only when id is separated
            if id in idsToUpdate:
                self.kalmanFilters[id].correct(x, y)
            else:
                self.kalmanFilters[id].correct(x, y, 0.8)


        # -------------- deleting old ids --------------------

        idsToDel = []
        for id, lastUpdate in self.idsLastUpdate.items():
            if self.frameCnt - lastUpdate > self.maxIdAge:
                idsToDel.append(id)

        for id in idsToDel:
            del(self.idsPositions[id])
            del(self.idsLastUpdate[id])
            del(idsHists[id])
            del(self.kalmanFilters[id])

        # ----------------- update memory --------------------

        self.lastIDs.append(dictids)
        self.lastBboxesCnt.append(bboxes_cnt)
        self.lastBboxes.append(bboxes)
        self.idsHists.append(idsHists)
        if len(self.lastBboxesCnt) > self.memSize:
            del (self.lastBboxesCnt[0])
            del (self.lastBboxes[0])
            del (self.lastIDs[0])
            del (self.idsHists[0])

        # -------------- delete not established ids ----------

        for boxid, ids in dictids.items():
            # if box contains one single id
            for id in ids:
                if id not in self.establishedIds:
                    # if the id was assigned to some box last time: increase the counter
                    # if id in flatten(self.lastIDs[-2].values()):
                    if id in self.idsFramesCnt.keys():
                        self.idsFramesCnt[id] += 1
                        if self.idsFramesCnt[id] == self.establishedThresh:
                            self.establishedIds.append(id)
                    # otherwise reset the counter
                    else:
                        self.idsFramesCnt[id] = 1
                    if self.idsFramesCnt[id] < self.establishedThresh:
                        dictids[boxid].remove(id)

        # ----------------- assign the oldest id to the box ---
        for boxid, ids in dictids.items():
            if len(ids) > 1:
                dictids[boxid] = self.__find_oldest_id(ids)

        return list(dictids.values())

    # def assignIDsContours(self, contours, frame):
    #     """
    #     assigns IDs to the contours
    #     :param contours:
    #     :param frame:
    #     :return:
    #     """
    #     contours = np.array(contours)
    #
    #     print(self.lastContours[-1])
    #     print(contours)
    #
    #     # compute distances
    #     distances = arbitrary_distance_matrix(contours, self.lastContours[-1], distance_contours)
    #
    #     # assigning boxID to each ID
    #     # for each previous bbox assign it's id to the closest new box if
    #     # the distance between them exceeds a threshold
    #     dictids = dict()
    #
    #     for i in range(len(contours)):
    #         dictids[i] = []
    #
    #     dist_T = distances.transpose()
    #     # assign previous IDs to new boxes
    #     i = 0
    #     for lastids, dist in zip(self.lastIDs[-1], dist_T):
    #         mindist = np.min(dist)
    #         if mindist <= self.distThreshold:
    #             # assign id of the closest next bbox
    #             for el in self.lastIDs[-1][i]:
    #                 if el not in dictids[np.argmin(dist)]:
    #                     dictids[np.argmin(dist)].append(el)
    #         i += 1
    #
    #     i = 0
    #     for contour, dist in zip(contours, distances):
    #         if len(dictids[i]) == 0:
    #             mindist = np.min(dist)
    #             # in case of splitting of a single blob assign to each blob all previous ids
    #             if mindist <= self.distThreshold:
    #                 for el in self.lastIDs[-1][np.argmin(dist)]:
    #                     if el not in dictids[i]:
    #                         dictids[i].append(el)
    #             else:
    #                 dictids[i].append(self.nextID)
    #                 self.kalmanFilters[self.nextID] = KalmanFilter()
    #                 self.nextID += 1
    #         i += 1
    #
    #     # update memory
    #     self.lastIDs.append(dictids)
    #     self.lastContours.append(contours)
    #     if len(self.lastIDs) > self.memSize:
    #         del (self.lastContours[0])
    #         del (self.lastIDs[0])
    #
    #     return list(dictids.values())
