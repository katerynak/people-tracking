#!/usr/bin/python3

import cv2
import numpy as np
from scipy.spatial import distance_matrix, distance
from distances import distance_contours, arbitrary_distance_matrix
from kalman_filter import KalmanFilter
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


class Obj_tracker(object):
    def __init__(self, frameWidth, frameHeight):
        # param init
        self.frameWidth = frameWidth
        self.frameHeight = frameHeight


        # memory of ids from previous frames
        # list of dictionaries of type blobID:IDs
        self.lastIDs = []
        self.lastIDs.append({0: [0]})

        # if bounding boxes are used
        # list of last bboxes, each blobID corresponds to the index in the list
        self.lastBboxes = []
        # first bbox : whole frame
        self.lastBboxes.append([0, 0, frameWidth, frameHeight])
        # centers of bboxes
        self.lastBboxesCnt = []
        self.lastBboxesCnt.append(np.array([[frameWidth // 2, frameHeight // 2]]))

        # if contours are used
        self.lastContours = []
        self.lastContours.append(np.array([[[[0, 0]], [[0, 719]], [[1279, 719]], [[1279, 0]]]]))

        # next available ID
        self.nextID = 1

        # number of bins for color histograms
        self.bins = 64
        # list of dictionaries of type ID:list of counts
        self.lastHists = []
        self.lastHists.append(np.zeros([self.bins, 1]))

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
        self.kalmanFilters[0] = KalmanFilter(frameWidth // 2, frameHeight // 2)
        x, y = self.lastBboxesCnt[-1][0]
        self.kalmanFilters[0].correct(x, y)

        # id: pos
        self.idsPositions = {}

        # id: lastFrame
        self.idsLastUpdate = {}
        self.idsLastUpdate[0] = 0

        self.maxIdAge = 8

        # number of frames to keep track of
        self.memSize = 2

        # distance threshold for id assignment, max distance a person can travel between 2 frames
        self.distThreshold1 = 70
        self.distThreshold = 130
        # trying out contours

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

    def similarity(self, h1, h2, id, box_x, box_y):
        """
        returns similarity of the box comparing to the statistics of the box corresponding to the target id
        in this function 2 similarity metrics are combined:
        1. color histogram similarity
        2. inverse distance similarity: 1/(predicted id position - box position)
        :param id:
        :param boxid:
        :return:
        """
        col_sim = self.histSimilarity(h1, h2)

        pos = self.kalmanFilters[id].lastPosition
        xp, yp = pos
        dist = distance.euclidean([box_x, box_y], [xp, yp])
        # dist = distance.euclidean([box_x], [xp])
        dist_sim = 1 / max(dist, 0.00001)
        print("----------------------------")
        print(col_sim)
        print(dist_sim)

        return col_sim + dist_sim*8
        # return dist_sim

    def similarity2(self, id, boxid, dist_weight = 0.0001):
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
        # dist = distance.euclidean([box_x], [xp])
        dist_sim = 1 / max(dist, 0.00001)
        if id==8:
            print("-----------------------")
            print(col_sim, dist_sim*dist_weight)
        return col_sim + dist_sim*dist_weight

    def most_similar(self, id, boxesIds):
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
            dists[i] = self.similarity2(id, boxId, dist_weight=4)
            i += 1
        print(dists)
        max_sim = boxesIds[np.argmax(dists)]
        return max_sim


    def __update_positions(self, bboxes_cnt):

        # idsToDel = []
        for id in self.kalmanFilters:

            self.idsPositions[id] = self.kalmanFilters[id].predict()
            # if self.idsPositions[id] is not None:
            #     # removing objects outside the scene
            #     if (self.idsPositions[id][0] < 1 or self.idsPositions[id][0] > self.frameWidth or
            #             self.idsPositions[id][1] < 1 or self.idsPositions[id][1] > self.frameHeight):
            #         idsToDel.append(id)

        # for id in idsToDel:
        #     del (self.idsPositions[id])
        #     del (self.kalmanFilters[id])


    def assignIDs(self, bboxes, frame_h):
        """
        assigns IDs to the bboxes
        :param bboxes:
        :param frame_h: hue component of the frame
        :return:
        """
        self.frameCnt += 1

        bboxes = np.array(bboxes)

        # compute centers of bounding boxes : x + w/2 , y + h//2
        bboxes_cnt = [list(bboxes[:, 0] + bboxes[:, 2] // 2), list(bboxes[:, 1] + bboxes[:, 3] // 2)]
        # transpose list of lists to get [x,y] couples
        bboxes_cnt = list(map(list, zip(*bboxes_cnt)))

        self.bboxes_cnt = bboxes_cnt

        # update id positions based on Kalman filter predictions
        self.__update_positions(bboxes_cnt)

        # compute color histograms of the current boxes
        hists = []
        for bbox in bboxes:
            subframe = frame_h[bbox[1]:bbox[1] + bbox[3], bbox[0]:bbox[0] + bbox[2]]
            # from 1 to 255 : we don't count the background mask info
            hist_h = cv2.calcHist([subframe], [0], None, [self.bins], [1, 230])
            hists.append(hist_h)

        self.hists = hists

        # compute distances
        distances = arbitrary_distance_matrix(bboxes_cnt, self.lastBboxesCnt[-1], distance.euclidean)

        distances_all = arbitrary_distance_matrix(bboxes_cnt, list(self.idsPositions.values()), distance.euclidean)
        # distances_all = arbitrary_distance_matrix(list(self.idsPositions.keys()), list(range(0, len(bboxes))),
        #                                           self.similarity2)


        dictids = dict()

        for i in range(len(bboxes)):
            dictids[i] = []

        # dist_T = distances_all
        dist_T = distances_all.transpose()

        # assign IDs to new boxes
        i = 0
        for id, dist in zip(self.idsPositions, dist_T):
            mindist = np.min(dist)
            if mindist <= self.distThreshold1:
                # assign id of the closest next bbox

                #select the closest 3 bboxes and assign an id to the most similar one
                orderedIdx = np.argsort(dist)
                ordered = np.sort(dist)
                closestBboxes = orderedIdx[ordered<self.distThreshold][:3]

                most_sim = self.most_similar(id, closestBboxes)
                # print(most_sim)
                # dictids[np.argmin(dist)].append(id)
                dictids[most_sim].append(id)
                self.idsLastUpdate[id] = self.frameCnt
            i += 1

        # assign new box to the closest box from the previous frame
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
                    x, y = bboxes_cnt[i]
                    self.kalmanFilters[self.nextID] = KalmanFilter(x, y)
                    self.idsLastUpdate[self.nextID] = self.frameCnt
                    self.nextID += 1

            i += 1

        # # assign new box to the closest id
        # i = 0
        # for bbox, dist in zip(bboxes, distances_all):
        #     mindist = np.min(dist)
        #     if mindist <= self.distThreshold:
        #         id = list(self.idsPositions.keys())[np.argmin(dist)]
        #         if id not in dictids[i]:
        #             dictids[i].append(id)
        #     else:
        #         dictids[i].append(self.nextID)
        #         x, y = bboxes_cnt[i]
        #         self.kalmanFilters[self.nextID] = KalmanFilter(x, y)
        #         self.nextID += 1
        #
        #     i += 1

        # ----------------- linear assignment of ids with multiple boxes ----------

        boxesToCheck = []
        idsToCheck = []

        for bboxIdx, ids in dictids.items():
            bboxiDs = getKeysWholeList(dictids, ids)
            if len(bboxiDs) > 1:
                if bboxiDs not in boxesToCheck:
                    boxesToCheck.append(bboxiDs)
                    idsToCheck.append(ids)

        for boxesIds, ids in zip(boxesToCheck, idsToCheck):

            cost_matrix = np.zeros([len(boxesIds), len(ids)])
            for (i, boxId) in enumerate(boxesIds):
                for (j, id) in enumerate(ids):
                    x, y = bboxes_cnt[boxId]
                    sim = self.similarity2(id, boxId)
                    # sim = self.similarity(self.idsHists[-1][id], hists[boxId], id, x, y)
                    # sim = self.histSimilarity(self.idsHists[id], hists[boxId])
                    cost_matrix[i, j] = -sim
                # if we have only one id, it will be assigned to the most similar box
            if len(ids) == len(boxesIds):
                sols = linear_assignment(cost_matrix)
                for sol in sols:
                    dictids[boxesIds[sol[0]]] = []
                for sol in sols:
                    dictids[boxesIds[sol[0]]].append(ids[sol[1]])

        # -------------- if one id appears in multiple boxes assign it to a single box

        # id : list of boxids associated with that id
        idsToCheck = {}
        ids = set(flatten(dictids.values()))
        for id in ids:
            idBoxes = getKeys(dictids, id)
            if len(idBoxes) > 1:
                idsToCheck[id] = idBoxes

        for id, boxids in idsToCheck.items():
            # boxid:similarity
            similarities = {}
            for boxid in boxids:
                x, y = bboxes_cnt[boxid]
                similarities[boxid] = self.similarity2(id, boxid)
                # similarities[boxid] = self.similarity(self.idsHists[-1][id], hists[boxid], id, x, y)
            winner = max(similarities, key=similarities.get)
            for boxid in boxids:
                dictids[boxid].remove(id)
            dictids[winner].append(id)

        # ------------------ assignment of new ids to boxes without ids -----------

        # check if some bboxes have lost all ids
        # for boxid, ids in dictids.items():
        #     if len(ids)==0:
        #         dictids[boxid].append(self.nextID)
        #         x, y = bboxes_cnt[boxid]
        #         self.kalmanFilters[self.nextID] = KalmanFilter(x, y)
        #         self.idsLastUpdate[self.nextID] = self.frameCnt
        #         self.nextID += 1

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
            idsHists[id] = hists[boxid]
            for phists in self.idsHists[::-1]:
                    for idx in phists:
                        if idx == id:
                            idsHists[id] += phists[id]
                            cnt += 1
            idsHists[id] /= cnt+1

        # self.idsHists.update(idsHists)

        # print(list(set(flatten(dictids.values()))))
        for id in list(set(flatten(dictids.values()))):
            x, y = bboxes_cnt[getKeys(dictids, id)[0]]
            # correct position only when id is separated
            if id in idsToUpdate:
                self.kalmanFilters[id].correct(x, y)
            else:
                self.kalmanFilters[id].correct(x, y, 0.9)

        # deleting old ids

        idsToDel = []
        for id, lastUpdate in self.idsLastUpdate.items():
            if self.frameCnt - lastUpdate > self.maxIdAge:
                idsToDel.append(id)
                print("{} to delete".format(id))

        for id in idsToDel:
            del(self.idsPositions[id])
            del(self.idsLastUpdate[id])
            del(idsHists[id])
            del(self.kalmanFilters[id])



        # update memory
        self.lastIDs.append(dictids)
        self.lastBboxesCnt.append(bboxes_cnt)
        self.lastBboxes.append(bboxes)
        self.lastHists.append(self.hists)
        self.idsHists.append(idsHists)
        if len(self.lastBboxesCnt) > self.memSize:
            del (self.lastBboxesCnt[0])
            del (self.lastBboxes[0])
            del (self.lastIDs[0])
            del (self.lastHists[0])
            del (self.idsHists[0])

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
