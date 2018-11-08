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

    def update(self, contours):
        if self.idx + 1 in self.frame_info:
            self.idx += 1
            self.predicted_avg += len(contours)
            self.truth_avg += self.frame_info[self.idx]
            self.frame_err_avg += abs(self.frame_info[self.idx] - len(contours))

    def print_stats(self):
        print("Frame error: %s" % (self.frame_err_avg/self.idx))
        print("Frame avg: %s" % (self.predicted_avg/self.idx))
        print("Frame truth avg: %s" % (self.truth_avg/self.idx))