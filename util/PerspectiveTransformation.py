import numpy as np
import cv2


class PerspectiveTransformation(object):
    """description of class"""

    def __init__(self, image_width, image_height):

        self.image_width = image_width
        self.image_height = image_height
        self.offset = 250

        transform_src = np.float32([(575,464),
                  (707,464), 
                  (258,682), 
                  (1049,682)])
        
        #np.float32([
        #    (132, 703),
        #    (540, 466),
        #    (740, 466),
        #    (1147, 703)])
            #np.float32([(600,450), (700,450), (200,720),(1150,720)]) # np.float32([ [580,450], [160,image_height], [1150,image_height], [740,450]])
        transform_dst = np.float32([(450,0),
                  (image_width-450,0),
                  (450,image_height),
                  (image_width-450,image_height)])
        
            #np.float32([
            #(transform_src[0][0] + self.offset, 720),
            #(transform_src[0][0] + self.offset, 0),
            #(transform_src[-1][0] - self.offset, 0),
            #(transform_src[-1][0] - self.offset, 720)])
            
            #np.float32([(300,0), (1000,0), (300,720), (1000,720)]) #np.float32([ [0,0], [0,image_height], [image_width,image_height], [image_width,0]])
        self.M = cv2.getPerspectiveTransform(transform_src, transform_dst)
        self.M_inv = cv2.getPerspectiveTransform(transform_dst, transform_src)

    def apply(self, undist_bin):
        return cv2.warpPerspective(undist_bin, self.M, (self.image_width,self.image_height), flags=cv2.INTER_LINEAR)

    def apply_inv(self, undist_bin):
        return cv2.warpPerspective(undist_bin, self.M_inv, (self.image_width,self.image_height), flags=cv2.INTER_LINEAR)


