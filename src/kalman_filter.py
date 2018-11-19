import cv2
import numpy as np


class KalmanFilter(object):
    def __init__(self, x, y):
        # internal state: x, y, velocity x, velocity y
        # prediction: x, y
        # self.kf = cv2.KalmanFilter(4, 2)
        # self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        # self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf = cv2.KalmanFilter(6, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0, 0, 0], [0, 1, 0, 1, 0, 0],
                                             [0, 0, 1, 0, 1, 0], [0, 0, 0, 1, 0, 1],
                                             [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]], np.float32)

        # number of times the filter was corrected
        self.corrections = 0
        self.threshold = 3

        # just for correction purposes, if the filter has predicted a value before it is not needed to predict
        # another time before correction
        self.toPredict = False
        self.lastPredicted = [x, y]
        self.lastPosition = [x, y]
        self.correct(x, y)

    def correct(self, x, y, weight = 1):

        self.lastPosition = [x, y]
        self.toPredict = True
        xp, yp = self.lastPredicted
        x = weight*x + (1-weight)*xp
        y = weight*y + (1-weight)*yp
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        self.corrections += 1

    def predict(self):
        self.toPredict = False
        # return None if the filter is still consider unreliable
        predicted = self.kf.predict()
        self.lastPredicted = [predicted[0][0], predicted[1][0]]
        if self.corrections > self.threshold:
            return self.lastPredicted
        else:
            return 0.8*np.array(self.lastPosition)+0.2*np.array(self.lastPredicted)