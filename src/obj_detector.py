import cv2


class Obj_detector(object):
    def __init__(self):
        # parameter initialization
        self.bbox_area_min = 100

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

    def detect_objects(self, mask):
        """
        detects object from a given binary mask, identify their contours
        :param mask:
        :return: list of contours of objects present in the binary mask
        """
        _, contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
        contours = self.select_contours(contours)
        return contours
