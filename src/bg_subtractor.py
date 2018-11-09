#!/usr/bin/python3

import cv2


class Bg_subtractor(object):
    def __init__(self, history=20, valThreshold=40, lr=0.01):
        # parameters initialization

        # MOG2 PARAMETERS
        self.history = history
        self.varThreshold = valThreshold
        self.lr = lr
        self.bgsb = cv2.createBackgroundSubtractorMOG2(self.history, self.varThreshold, detectShadows=False)

        # morphological transformations kernels
        self.open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        self.close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

    def morph_trans(self, frame, open_iter=1, close_iter=5):
        """
        morphological transformations on a given frame
        :param frame:
        :param open_iter: opening kernel
        :param close_iter: closing kernel
        :return:
        """
        ret = frame.copy()
        ret = cv2.morphologyEx(ret, cv2.MORPH_OPEN, self.open, iterations=open_iter)
        ret = cv2.morphologyEx(ret, cv2.MORPH_CLOSE, self.close, iterations=close_iter)
        return ret

    def rgb2h(self, frame):
        """
        given a frame in rgb space the function changes color space and extracts one channel hue
        info
        :param frame:
        :return:
        """
        # hsv contains hue and saturation components that are not dependent on the light info
        hsv = frame.copy()
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # some bluring added in order to make the background more uniform

        h = cv2.blur(h, (10, 5))

        h[h>127] = 0

        return h

    def fg_mask(self, frame):
        """
        given a frame returns a binary mask for foreground filtering
        :param frame: single channel image
        :return:
        """
        foreground = frame.copy()
        foreground = self.rgb2h(foreground)
        foreground = self.bgsb.apply(foreground, learningRate=self.lr)
        foreground = self.morph_trans(foreground)

        return foreground


if __name__ == "__main__":
    # background subtraction model
    bg_sub = Bg_subtractor()

    cap = cv2.VideoCapture('../pedestrians.mp4')
    width = int(cap.get(3))
    height = int(cap.get(4))

    # create a named windows and move it
    cv2.namedWindow('video')
    cv2.moveWindow('video', 70, 30)

    next_frame, frame = cap.read()
    play = True

    list = []
    cnt = 0

    den_frames = 5

    hd = frame

    while cap.isOpened():
        if play:

            # extract the foreground mask
            mask = bg_sub.fg_mask(frame)

            # apply it and obtain an rgb foreground
            col_foreground = cv2.bitwise_and(frame, frame, mask=mask)

            h = bg_sub.rgb2h(frame)

            # h = cv2.fastNlMeansDenoising(h, searchWindowSize = 5)

            cv2.imshow('mask', col_foreground)

            h = cv2.blur(h, (10, 5))

            cv2.imshow('h', h)

            next_frame, frame = cap.read()
            cnt += 1

        # q for exit, space for pause
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == 0x20:
            play = not play

    cap.release()
    cv2.destroyAllWindows()
