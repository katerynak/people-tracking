#!/usr/bin/python3

import cv2
import numpy as np
from scipy.spatial import distance
from distances import distance_contours, arbitrary_distance_matrix
from kalman_filter import KalmanFilter
import heapq
import math

# function to flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


def getKeys(mydict, value):
    """
    given a dictionary value: a [sub]list or a value, returns a list of keys of items containing
    that value or sublist
    :param dict: dicitonary
    :param value: value or [sub]list of values
    :return:
    """
    ret = []

    for key, val in mydict.items():
        if value in val:
            ret.append(key)

    return ret


def getKeysWholeList(mydict, value):
    """
    given a dictionary value: list and a value, returns a list of keys of items containing
    that exact value
    :param dict: dicitonary
    :param value: value
    :return:
    """
    ret = []

    for key, val in mydict.items():
        if value == val:
            ret.append(key)

    return ret


class Obj_tracker(object):
    def __init__(self, frameWidth, frameHeight):
        """
        tracker initialization function
        :param frameWidth: width in pixels
        :param frameHeight: height in pixels
        """
        # param init
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight

        # -------------------- information about previous frame[s] ---------------

        # memory of ids from previous frames, for each bounding box a list of ids is associated
        # list of dictionaries of type {bboxID:IDs}
        self.lastIDs = []
        # initialization of previous history in order to have a sort of memory for
        # processing of the first frame
        self.lastIDs.append({})
        self.lastIDs.append({})

        self.lastIDsPositions = []
        self.lastIDsPositions.append({})

        self.lastIDsPositionsMem = 15

        # if bounding boxes are used
        # list of last bboxes, each bboxID corresponds to the index in the list
        self.lastBboxes = []
        # centers of bboxes
        self.lastBboxesCnt = []

        # -------------------- variables used for ids assignment ------------------

        # next available ID
        self.nextID = 1

        # number of bins for color histograms
        self.bins = 64

        # list of histograms of single ids, {id: hist}
        self.idsHists = []
        self.idsHists.append({})

        # histograms of current bounding boxes of the current frame
        # {bboxID: hist}
        self.hists = {}

        # positions of the bounding boxes of the current frame
        # {bboxID: [x, y]}
        self.bboxes_cnt = {}

        # dictionary of kalman filters of single ids
        # {id: KalmanFilter instance}
        self.kalmanFilters = {}

        # correction weight for ids assigned to bboxes with multiple ids
        self.correction_weight_multiple = 0.9
        # correction weight for ids assigned to bboxes which touch frame borders
        self.correction_weight_out = 0.4

        # positions of ids predicted by the Kalman filters
        # {id: [x,y]}
        self.idsPositions = {}

        # velocities of ids predicted by the Kalman filters
        # {id: [vel_x, vel_y]}
        self.idsVelocities = {}

        # number of candidate bounding box for the id assignment
        self.candidates = 4

        # euclidean distance threshold for id assignment, max distance a person can travel between 2 frames
        self.distThreshold = 80

        # number of frames to keep track of
        self.memSize = 15

        # weights of contributions of color, distance and velocity to the similarity metric
        self.col_weight = 0.80
        self.dist_weight = 0.2
        self.vel_weight = 0

        # if the similarity of the closest bounding box and an id is less then sim_threshold,
        # id will not be assigned to that box
        self.sim_threshold = 0.1

        # maximum people density : problem dependent variable
        # if bounding box has area <= bbox_area_min_2 it cannot contain 2 ids since too small
        # only one id will be assigned to that bbox
        self.bbox_area_min_4 = 8000
        self.bbox_area_min_2 = 3000

        # -------------------- info for detecting ids last usage / age ------------

        # last frame in which ad id was assigned to some bounding box
        # {id: lastFrame}
        self.idsLastUpdate = {}

        # first frame in which ad id was assigned to some bounding box
        # {id: fistFrame}
        self.idsFirstFrame = {}
        # counter for subsequent frames, for identifying temporary and well established ids
        # id: subsequent frames counter
        self.idsFramesCnt = {}

        # threshold for an id to be considered as established id ( useful for cleaning of
        # temporal bounding boxes (due to body splitting for example))
        self.establishedThresh = 5

        # list of established ids
        self.establishedIds = []

        # if an id was not assigned to any bounding box for a maximum number of frames it will be deleted
        self.maxIdAge = 15

        # frame counter
        self.frameCnt = 0

    def histSimilarity(self, hist1, hist2, method=cv2.HISTCMP_INTERSECT, normalize=True):
        """
        computes color similarity between hist1 and hist2
        hist1 and hist2 must be of the same length
        other methods are described here:
        https://docs.opencv.org/2.4/modules/imgproc/doc/histograms.html?highlight=comparehist#comparehist
        with the following notation for opencv3:
        cv2.HISTCMP_CORREL
        cv2.HISTCMP_CHISQR
        cv2.HISTCMP_INTERSECT
        cv2.HISTCMP_BHATTACHARYYA
        :param hist1: list containing bin counts of img1
        :param hist2: list containing bin counts of img2
        :return: float containing similarity of two histograms
        """
        # hist normalization
        if normalize:
            h1 = hist1 / np.sum(hist1)
            h2 = hist2 / np.sum(hist2)

        return cv2.compareHist(h1, h2, method=method)

    def similarity(self, id, boxID, col_weight = 0.6, dist_weight = 0.3, vel_weight = 0.1):
        """
        returns similarity of the bounding box having index boxID and the id
        in this function 3 similarity metrics are combined:
        1. color histogram similarity
        2. inverse distance similarity: 1 - (predicted id position - box position)/maxdistance
        3. velocity similarity: cosine similarity between id velocity and a vector
                                represented by (candidate position - id position)
        :param id: id of a pedestrian
        :param boxID: id of a bounding box
        :params col_weight, dist_weight, vel_weight: weights of color, distance and velocity contributions;
                                                     should sum to 1
        :return: similarity of the bounding box and id, float ranging between 0 and 1
        """
        h1 = self.hists[boxID]
        if len(self.idsHists[-1].keys()) == 0:
            col_sim = 0
        else:
            h2 = self.idsHists[-1][id]
            col_sim = self.histSimilarity(h1, h2)
        box_x, box_y = self.bboxes_cnt[boxID]

        pos = self.kalmanFilters[id].lastPosition
        xp, yp = pos

        # distance in pixels
        dist = distance.euclidean([box_x, box_y], [xp, yp])
        # distance normalization
        dist = dist / self.distThreshold
        dist_sim = 1 - dist
        dist_sim += dist_sim**5

        # evaluating last x and y predicted positions
        xp -= self.kalmanFilters[id].lastPredictedVelocity[0]
        yp -= self.kalmanFilters[id].lastPredictedVelocity[1]

        # velocity similarity : calculated as cosine between id's velocity vector and a vector
        #                       represented by (candidate position - id position)
        vel_sim = 1 - distance.cosine(self.kalmanFilters[id].lastPredictedVelocity,
                                      [box_x - xp, box_y - yp])

        if math.isnan(vel_sim):
            vel_sim = 0
            # if no information is provided about the velocity, redistribute weights to color and distance info
            col_weight += vel_weight/2
            dist_weight += vel_weight/2

        return col_sim*col_weight + dist_sim*dist_weight + vel_sim*vel_weight

    def most_similar(self, id, boxesIDs, sim_threshold = 0.1):
        """
        given an id and boxIDs of the closest bounding boxes
        returns the most similar to the pedestrian (according to color + euclidean distances + velocity) box id
        :param id: id of the pedestrian
        :param boxesIDs: candidate bounding boxes
        :return: id of the most similar bounding box
        """
        dists = np.zeros(len(boxesIDs))
        i = 0
        for boxID in boxesIDs:
            dists[i] = self.similarity(id, boxID, self.col_weight, self.dist_weight, self.vel_weight)
            i += 1
        if np.max(dists > sim_threshold):
            max_sim = boxesIDs[np.argmax(dists)]
        else:
            max_sim = -1
        return max_sim

    def __delele_id(self, id):
        """
        deletes id from the memory
        :param id: id index to be deleted
        :return:
        """
        del (self.idsPositions[id])
        del (self.idsLastUpdate[id])
        del (self.kalmanFilters[id])

    def __update_positions(self):
        """
        updates positions and velocities of the pedestrians according to the kalman filters predictions
        :return:
        """
        for id in self.kalmanFilters:
            self.idsPositions[id] = self.kalmanFilters[id].predict()
            self.idsVelocities[id] = self.kalmanFilters[id].lastPredictedVelocity

    def __delete_exited(self):
        """
        deletes ids exited from the scene
        :return:
        """
        idsToDel = []
        for id in self.idsPositions:
            if self.idsPositions[id][0] < 0 or self.idsPositions[id][0] > self.frameWidth:
                # if an id is old (new ids may be wrongly located outside the scene)
                if self.frameCnt - self.idsFirstFrame[id] > 50:
                    idsToDel.append(id)

        for id in idsToDel:
            self.__delele_id(id)

    def __find_oldest_id(self, ids, n=1):
        """
        among provided ids returns the n oldest ones
        :param ids: list of ids
        :param n: int, number of the oldest ids to be found
        :return: list of n oldest ids
        """
        firstFrames = {}
        for id in ids:
            firstFrames[id] = self.idsFirstFrame[id]

        return heapq.nsmallest(n, firstFrames, key=firstFrames.get)

    def assignIDs(self, bboxes, frame):
        """
        assigns IDs to the bounding boxes
        :param bboxes: list of bounding boxes of moving objects in the frame
        :param frame: one component only, used for computing color histograms
        :return: list of IDs corresponding to the input list of the bounding boxes
        """

        # increase frame counter
        self.frameCnt += 1

        # --------------- compute bbox coordinates --------------------------
        bboxes = np.array(bboxes)

        # compute centers of bounding boxes : x + w/2 , y + h/2
        bboxes_cnt = [list(bboxes[:, 0] + bboxes[:, 2] // 2), list(bboxes[:, 1] + bboxes[:, 3] // 2)]
        # transpose list of lists to get [x,y] couples
        bboxes_cnt = list(map(list, zip(*bboxes_cnt)))

        self.bboxes_cnt = bboxes_cnt

        # ---------------- updating kalman predictions ----------------------
        # update id positions based on Kalman filter predictions
        self.__update_positions()
        self.__delete_exited()

        # -------- computing current color histograms of all bboxes ---------
        # compute color histograms of the current boxes
        hists = []
        for bbox in bboxes:
            subframe = frame[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            # from 1 to 255 : we don't count the background mask info
            hist_h = cv2.calcHist([subframe], [0], None, [self.bins], [1, 230])
            hists.append(hist_h)

        self.hists = hists

        # ------------------- assign IDs to new boxes ----------------------------
        distances = arbitrary_distance_matrix(list(self.idsPositions.values()), bboxes_cnt, distance.euclidean)

        dictids = dict()

        for i in range(len(bboxes)):
            dictids[i] = []

        # assign id of the closest most similar bbox
        for id, dist in zip(self.idsPositions, distances):
            mindist = np.min(dist)
            if mindist <= self.distThreshold:
                # select the closest bboxes and assign an id to the most similar one
                orderedIdx = np.argsort(dist)
                ordered = np.sort(dist)
                closestBboxes = orderedIdx[ordered < self.distThreshold][:self.candidates]
                most_sim = self.most_similar(id, closestBboxes, self.sim_threshold)
                if most_sim != -1:
                    dictids[most_sim].append(id)
                    self.idsLastUpdate[id] = self.frameCnt

        # ------------------ assignment of new ids to boxes without ids -----------

        # check if some bboxes have lost all ids
        for boxID, ids in dictids.items():
            if len(ids)==0:
                dictids[boxID].append(self.nextID)
                x, y = bboxes_cnt[boxID]
                self.kalmanFilters[self.nextID] = KalmanFilter(x, y)
                self.idsLastUpdate[self.nextID] = self.frameCnt
                self.idsFirstFrame[self.nextID] = self.frameCnt
                self.nextID += 1

        # ------------------ a box can contain max 4 persons, if it contains more, cut out the youngest-
        # if a box contains 4 persons or more check also it's dimension, if it's too small cut the number of ids

        for boxID, ids in dictids.items():
            if len(ids) > 4:
                dictids[boxID] = self.__find_oldest_id(ids, n=4)
                idxToDel = list(set(ids) - set(dictids[boxID]))
                for idx in idxToDel:
                    del (self.idsPositions[idx])
                    del (self.idsLastUpdate[idx])
                    del (self.kalmanFilters[idx])

        # -------------- if a box contains more then one id check if its area is not too large:
        # -------------- if box's area is less then the minimum for 2 ids, keep the oldest id only
        # -------------- if box's area is less then the min for 4 ids, keep the two oldest ids only

        for boxID, ids in dictids.items():
            if len(ids) > 1 and len(ids) < 4:
                if bboxes[boxID][2] * bboxes[boxID][3] < self.bbox_area_min_2:
                    dictids[boxID] = self.__find_oldest_id(ids, n=1)
                    idxToDel = list(set(ids) - set(dictids[boxID]))
                    for idx in idxToDel:
                        self.__delele_id(idx)
            if len(ids) == 4:
                if bboxes[boxID][2] * bboxes[boxID][3] < self.bbox_area_min_4:
                    dictids[boxID] = self.__find_oldest_id(ids, n=2)
                    idxToDel = list(set(ids) - set(dictids[boxID]))
                    for idx in idxToDel:
                        self.__delele_id(idx)

        # ------------------ updating histograms only of bboxes with single id ------------------
        # decide which color histograms to update: only an id assigned to a single bbox
        # and in case when it is the only id of that bbox
        # updating histograms, contains couples id:boxID
        idsToUpdate = {}
        for boxID, ids in dictids.items():
            # if box contains one single id
            if len(ids) == 1:
                # check if that id is contained only in one box
                if len(getKeys(dictids, ids[0])) == 1:
                    idsToUpdate[ids[0]] = boxID

        idsHists = self.idsHists[-1]
        # update the histograms of selected ids
        for id, boxID in idsToUpdate.items():
            cnt = 0
            idsHists[id] = hists[boxID]
            # averaging with previous histograms
            for phists in self.idsHists[::-1]:
                    for idx in phists:
                        if idx == id:
                            idsHists[id] += phists[id]
                            cnt += 1
            idsHists[id] /= cnt+1

        # ------------------- correcting Kalman filters: for box with single id, ----------------
        # uses its position, for those with multiple ids uses
        # a combination between predicted position and weighted box position

        for id in list(set(flatten(dictids.values()))):
            bboxID = getKeys(dictids, id)[0]
            x, y = bboxes_cnt[bboxID]
            # correct position only when id is separated
            if id in idsToUpdate:
                if (bboxes[bboxID][0]+bboxes[bboxID][2] >= self.frameWidth) or (bboxes[bboxID][0] <= 0):
                    if self.frameCnt - self.idsFirstFrame[id] > 50:
                        self.kalmanFilters[id].correct(x, y, self.correction_weight_out)
                    else:
                        self.kalmanFilters[id].correct(x, y)
                else:
                    self.kalmanFilters[id].correct(x, y)
            else:
                self.kalmanFilters[id].correct(x, y, self.correction_weight_multiple)

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

        # ----------------- updating memory --------------------

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

        for boxID, ids in dictids.items():
            # if box contains one single id
            for id in ids:
                # if self.frameCnt == 2:
                #     print(id)
                #     self.establishedIds.append(id)
                if id not in self.establishedIds:
                    # if the id appeared last n times
                    if id in self.idsFramesCnt.keys():
                        self.idsFramesCnt[id] += 1
                        if self.idsFramesCnt[id] == self.establishedThresh:
                            self.establishedIds.append(id)
                    # otherwise reset the counter
                    else:
                        self.idsFramesCnt[id] = 1
                    if self.idsFramesCnt[id] < self.establishedThresh:
                        dictids[boxID].remove(id)

        # ----------------- assign the oldest id to the box ---
        for boxID, ids in dictids.items():
            if len(ids) > 1:
                dictids[boxID] = self.__find_oldest_id(ids)

        # updating last ids positions
        lastIDsPositions = {}
        for boxID, id in dictids.items():
            if len(id) > 0:
                lastIDsPositions[id[0]] = bboxes_cnt[boxID]

        self.lastIDsPositions.append(lastIDsPositions)

        if len(self.lastIDsPositions) > self.lastIDsPositionsMem:
            del(self.lastIDsPositions[0])

        return list(dictids.values())