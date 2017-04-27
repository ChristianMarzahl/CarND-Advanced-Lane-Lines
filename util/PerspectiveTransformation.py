import numpy as np
import cv2


class PerspectiveTransformation(object):
    """description of class"""

    def __init__(self, image_width, image_height):

        self.image_width = image_width
        self.image_height = image_height
        self.offset = 450

        transform_src = np.float32([(570,460),
                  (710,460), 
                  (225,690), 
                  (1070,680)])
        
        transform_dst = np.float32([(self.offset,0),
                  (image_width-self.offset,0),
                  (self.offset,image_height),
                  (image_width-self.offset,image_height)])

        self.M = cv2.getPerspectiveTransform(transform_src, transform_dst)
        self.M_inv = cv2.getPerspectiveTransform(transform_dst, transform_src)

    def apply(self, undist_bin):
        return cv2.warpPerspective(undist_bin, self.M, (self.image_width,self.image_height), flags=cv2.INTER_LINEAR)

    def apply_inv(self, undist_bin):
        return cv2.warpPerspective(undist_bin, self.M_inv, (self.image_width,self.image_height), flags=cv2.INTER_LINEAR)


