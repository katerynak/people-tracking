import numpy as np
from operator import itemgetter

# function to flatten list of lists
flatten = lambda l: [item for sublist in l for item in sublist]

class Statistics(object):
    def __init__(self, groundTruthFile):

        # ground truth parsing
        self.frame_info = dict()
        with open(groundTruthFile, "r") as f:
            for line in f:
                frame_num, id, x, y = line.split(',')
                frame_num = int(frame_num)
                self.frame_info[frame_num] = self.frame_info.get(frame_num, 0) + 1

        # stat parameters initialization
        self.idx = 0
        self.frame_err_avg = 0
        self.predicted_avg = 0
        self.truth_avg = 0

        # size in pixel of the gate for counting entering and exiting persons
        self.gate_size = 60

        self.minExitingAge = 30

    def update(self, contours):
        """
        if the frame is present in ground truth file the stats about performances are updated
        :param contours: list of contours in current frame
        :return:
        """
        if self.idx + 1 in self.frame_info:
            self.idx += 1
            self.predicted_avg += len(contours)
            self.truth_avg += self.frame_info[self.idx]
            self.frame_err_avg += abs(self.frame_info[self.idx] - len(contours))

    def count_entering_exiting(self, bboxes_cnt, bboxes_ids, ids_velocity, frame_width, idsFirstFrame, currFrame):
        """
        counts pedestrians entering and exiting the scene, taking into account
        pedestrians in the gate zone and information about their velocity
        :param bboxes_cnt: positions of the centers of bounding boxes
        :param bboxes_ids: ids of bounding boxes
        :param ids_velocity: velocity of pedestrians
        :param ids_firstFrame: first frame in which the id was seen in the scene, useful for identifying new ids
        :param curr_frame:
        :return: entering pedestrians counts
        """
        # selecting pedestrians inside the gate

        bboxes_cnt = np.array(bboxes_cnt)

        left_gate_idx = np.argwhere(bboxes_cnt[:, 0] < self.gate_size)

        left_gate_idx = flatten(left_gate_idx)

        if len(left_gate_idx) > 0:

            left_gate_ids = itemgetter(*left_gate_idx)(bboxes_ids)
        else:
            left_gate_ids = []

        right_gate_idx = np.argwhere(bboxes_cnt[:, 0] > frame_width-self.gate_size)
        right_gate_idx = flatten(right_gate_idx)

        entering = 0
        exiting = 0

        for id in left_gate_ids:
            id = np.atleast_1d(id)
            if len(id) == 0:
                entering += 1
            elif ids_velocity[id[0]][0] >= 0:
                entering += 1
            elif currFrame - idsFirstFrame[id[0]] < self.minExitingAge:
                entering += 1
            else:
                # print("id {} vel {}".format(id, ids_velocity[id[0]][0]))
                exiting += 1

        if len(right_gate_idx) > 0:

            right_gate_ids = itemgetter(*right_gate_idx)(bboxes_ids)
        else:

            right_gate_ids = []

        for id in right_gate_ids:
            id = np.atleast_1d(id)
            if len(id) == 0:
                entering += 1
            elif ids_velocity[id[0]][0] <= 0:
                entering += 1
            elif currFrame - idsFirstFrame[id[0]] < self.minExitingAge:
                entering += 1
            else:
                exiting += 1

        return entering, exiting

    def print_stats(self):
        print("Frame error: %s" % (self.frame_err_avg/self.idx))
        print("Frame avg: %s" % (self.predicted_avg/self.idx))
        print("Frame truth avg: %s" % (self.truth_avg/self.idx))

    def get_curr_truth_counts(self):
        """
        returns number of current ground truth counts
        :return:
        """
        return self.frame_info[self.idx]