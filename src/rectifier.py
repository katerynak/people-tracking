#!/usr/bin/python3

import cv2
import numpy as np


class Rectifier(object):
    def __init__(self, width, height, shift=100, up=True, central_zoom = 50):
        self.width = width
        self.height = height
        self.shift = shift

        # source and destination points left/right

        if up:
            pts_src_l = np.array([[0, 0], [0, height], [int(width / 2), 0], [int(width / 2), height]])

            pts_dst_l = np.array([[-shift, -shift], [-shift, height+shift],
                                  [int(width / 2), -central_zoom], [int(width / 2), height+central_zoom]])

            pts_src_r = np.array([[int(width / 2), 0], [int(width / 2), height], [width, 0], [width, height]])

            pts_dst_r = np.array([[0, -central_zoom], [0, height+central_zoom],
                                  [int(width/2)+shift, -shift], [int(width/2)+shift, height+shift]])

            pts_src_r_back = np.array([[int(width / 2), -central_zoom], [int(width / 2), height+central_zoom],
                                       [width + shift, -shift], [width + shift, height + shift]])
            pts_dst_r_back = pts_src_l

        else:

            pts_src_l = np.array([[0, 0], [0, height], [int(width / 2), 0], [int(width / 2), height]])

            pts_dst_l = np.array([[shift, shift], [shift, height - shift], [int(width / 2), 0],
                                       [int(width / 2), height]])

            pts_src_r = np.array([[int(width / 2), 0], [int(width / 2), height], [width, 0], [width, height]])


            pts_dst_r = np.array([[0, 0], [0, height],
                                [int(width / 2) - shift, shift], [int(width / 2) - shift, height - shift]])

            pts_src_r_back = np.array([[int(width / 2), 0], [int(width / 2), height],
                                [width - shift, shift], [width - shift, height - shift]])
            pts_dst_r_back = pts_src_l

        # calculate homography
        self.h_l, _ = cv2.findHomography(pts_src_l, pts_dst_l)
        self.h_r, _ = cv2.findHomography(pts_src_r, pts_dst_r)

        # calculate back transform
        self.h_back_l, _ = cv2.findHomography(pts_dst_l, pts_src_l)
        self.h_back_r, _ = cv2.findHomography(pts_src_r_back, pts_dst_r_back)

    def rectify(self, frame):
        """
        given an image change its holographic perspective: "rectifying" the image
        :param frame:
        :return:
        """
        im_out_l = cv2.warpPerspective(frame, self.h_l, (int(self.width / 2), self.height))
        im_out_r = cv2.warpPerspective(frame, self.h_r, (int(self.width / 2), self.height))

        vis = np.concatenate((im_out_l, im_out_r), axis=1)

        return vis

    def projection(self, vis):
        """
        given a "rectified" image change its holographic perspective: back to "projection"
        :param vis:
        :return:
        """
        im_orig_l = cv2.warpPerspective(vis, self.h_back_l, (int(self.width / 2), self.height))
        im_orig_r = cv2.warpPerspective(vis, self.h_back_r, (int(self.width / 2), self.height))

        frame = np.concatenate((im_orig_l, im_orig_r), axis=1)

        return frame


if __name__ == "__main__":

    cap = cv2.VideoCapture('../pedestrians.mp4')
    width = int(cap.get(3))
    height = int(cap.get(4))

    rect = Rectifier(width, height, 100)

    # create a named windows and move it
    cv2.namedWindow('video')
    cv2.moveWindow('video', 70, 30)

    cv2.namedWindow('rectified')
    cv2.moveWindow('rectified', 70, 30)

    cv2.namedWindow('projection')
    cv2.moveWindow('projection', 70, 30)

    next_frame, frame = cap.read()
    play = True

    while cap.isOpened():
        if play:
            rectified = rect.rectify(frame)

            cv2.circle(rectified, (200, 200), 10, (0, 255, 0), 2)

            proj = rect.projection(rectified)

            # display the image
            cv2.imshow('rectified', rectified)

            cv2.imshow('projection', proj)

            cv2.imshow('video', frame)

            next_frame, frame = cap.read()

            # q for exit, space for pause
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == 0x20:
            play = not play

    cap.release()
    cv2.destroyAllWindows()