import numpy as np
import cv2

class ThresholdRoadImage(object):
    """description of class"""

    def __init__(self, sx_thresh = (20,100), s_thresh = (170,255)):

        self.sx_thresh = sx_thresh
        self.s_thresh = s_thresh

    def compute_threshold(self, img):
        
        #combined_binary = self.color_and_gradient(img)

        # HLS L-channel Threshold (using default parameters)
        img_LThresh = self.hls_lthresh(img)

        # Lab B-channel Threshold (using default parameters)
        img_BThresh = self.lab_bthresh(img)
    
        # Combine HLS and Lab B channel thresholds
        combined_binary = np.zeros_like(img_BThresh)
        combined_binary[(img_LThresh == 1) | (img_BThresh == 1)] = 255

        return combined_binary

    def color_and_gradient(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
        # be beneficial to replace this channel with something else.
        #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
        
        binary_output = np.zeros_like(sxbinary)
        binary_output[(sxbinary == 1) | (s_binary == 1)] = 1
        return binary_output

    def lab_bthresh(self, img, thresh=(190,255)): # 190
        # 1) Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_b = lab[:,:,2]
        # don't normalize if there are no yellows in the image
        if np.max(lab_b) > 175:
            lab_b = lab_b*(255/np.max(lab_b))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(lab_b)
        binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    def hls_lthresh(self, img, thresh=(220, 255)):# 220
        # 1) Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_l = hls[:,:,1]
        hls_l = hls_l*(255/np.max(hls_l))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(hls_l)
        binary_output[(hls_l > thresh[0]) & (hls_l <= thresh[1])] = 1
        # 3) Return a binary image of threshold result
        return binary_output