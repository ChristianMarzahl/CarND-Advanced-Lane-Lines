import cv2
import numpy as np
import glob
import os
import pickle

from util import ThresholdRoadImage
from util import PerspectiveTransformation
from util import CamCalibration

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y
calibration_images = glob.glob('camera_cal/calibration*.jpg')

cam_calibration = None
calibration_file = "camera_calibration.pickle"

if os.path.exists(calibration_file):
    cam_calibration = CamCalibration.CameraCalibration(calibration_file)
else: 
    Camera_Calibration.calculate_and_save_calibrations_matrix(calibration_images,nx,ny,calibration_file)
    cam_calibration = CamCalibration.CameraCalibration(calibration_file)

img = cv2.imread('camera_cal/calibration10.jpg')
undistorted_image = cam_calibration.undistort_image(img)

# load image
test_image = cv2.imread("test_images/straight_lines1.jpg")

# undistort image 
undist_iamge = cam_calibration.undistort_image(test_image)

# threshold image 
thresholdRoadImage = ThresholdRoadImage.ThresholdRoadImage()
threshold_image = thresholdRoadImage.compute_threshold(undist_iamge)

cv2.imshow("Hallo",threshold_image)
cv2.waitKey(0)

# perspective transformation
shape = threshold_image.shape[::-1] # (width,height)
w = shape[0]
h = shape[1]
perspectiveTransformation = PerspectiveTransformation.PerspectiveTransformation(w,h)
warped_image = perspectiveTransformation.apply(threshold_image)

cv2.imshow("Hallo",warped_image)
cv2.waitKey(0)

