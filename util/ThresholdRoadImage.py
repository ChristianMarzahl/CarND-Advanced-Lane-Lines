import numpy as np
import cv2

class ThresholdRoadImage(object):
    """description of class"""

    def __init__(self, sx_thresh = (20,100), s_thresh = (170,255)):

        self.sx_thresh = sx_thresh
        self.s_thresh = s_thresh

    def compute_threshold(self, img):
        
        #combined_binary = self.color_and_gradient(img)

        ## HLS hls_thresh
        img_HLS_Thresh = self.hls_thresh(img)

        ## Lab Threshold 
        img_LAB_Thresh = self.lab_thresh(img)
    
        ## Combine HLS and LAB channel thresholds
        combined_binary = np.zeros_like(img_LAB_Thresh)
        combined_binary[(img_HLS_Thresh == 1) | (img_LAB_Thresh == 1)] = 255

        return combined_binary

    def color_and_gradient(self, img, s_thresh=(170, 255), sx_thresh=(20, 100)):
        img = np.copy(img)
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:,:,1]
        s_channel = hsv[:,:,2]
        # Sobel x
        sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
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
        binary_output[(sxbinary == 1) | (s_binary == 1)] = 255
        return binary_output

    def lab_thresh(self, img, thresh=(190,255), channel = 2): # 190
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_c = lab[:,:,channel]

        lab_c = lab_c * (255 / lab_c.max())
        #lab_c = cv2.equalizeHist(lab_c)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #lab_c = clahe.apply(lab_c)

        #cv2.normalize(lab_c,lab_c,0,255,cv2.NORM_MINMAX)


        #lab_c = lab_c*(255/np.max(lab_c))
        # don't normalize if there are no yellows in the image
        #if np.max(lab_b) > 175:
        #    lab_b = lab_b*(255/np.max(lab_b))
        # 2) Apply a threshold to the L channel
        binary_output = np.zeros_like(lab_c)
        binary_output[((lab_c > thresh[0]) & (lab_c <= thresh[1]))] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    def hls_thresh(self, img, thresh=(220, 255), channel = 1):# 220
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_c = hls[:,:,channel]

        hls_c = hls_c * (255 / hls_c.max())

        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        #hls_c = clahe.apply(hls_c)

        #hls_c *= 255 / hls_c.max()
        #cv2.normalize(hls_c,hls_c,0,255,cv2.NORM_MINMAX)

        # Apply a threshold to the channel
        binary_output = np.zeros_like(hls_c)
        binary_output[(hls_c > thresh[0]) & (hls_c <= thresh[1])] = 1
        # Return a binary image of threshold result
        return binary_output