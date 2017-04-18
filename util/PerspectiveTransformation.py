import numpy as np
import cv2


class PerspectiveTransformation(object):
    """description of class"""

    def __init__(self, image_width, image_height):

        self.image_width = image_width
        self.image_height = image_height

        transform_src = np.float32([ [580,450], [160,image_height], [1150,image_height], [740,450]])
        transform_dst = np.float32([ [0,0], [0,image_height], [image_width,image_height], [image_width,0]])
        M = cv2.getPerspectiveTransform(transform_src, transform_dst)
        self.M = M

    def apply(self, undist_bin):
        return cv2.warpPerspective(undist_bin, self.M, (self.image_width,self.image_height))




