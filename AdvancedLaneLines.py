import cv2
import numpy as np
import glob
import os
import pickle

from util.ThresholdRoadImage import ThresholdRoadImage
from util.PerspectiveTransformation import PerspectiveTransformation
from util.CamCalibration import  CameraCalibration
from util.LaneDetector import LaneDetector

# https://github.com/jeremy-shannon/CarND-Advanced-Lane-Lines

nx = 9 # the number of inside corners in x
ny = 6 # the number of inside corners in y
calibration_images = glob.glob('camera_cal/calibration*.jpg')

cam_calibration = None
calibration_file = "camera_calibration.pickle"

if os.path.exists(calibration_file):
    cam_calibration = CameraCalibration(calibration_file)
else: 
    Camera_Calibration.calculate_and_save_calibrations_matrix(calibration_images,nx,ny,calibration_file)
    cam_calibration = CameraCalibration(calibration_file)

    
# threshold image 
thresholdRoadImage = ThresholdRoadImage()

test_images_paths = glob.glob('test_images/test*.jpg')

# perspective transformation
perspectiveTransformation = PerspectiveTransformation(1280,720)
lineDetector = LaneDetector(cam_calibration,perspectiveTransformation,thresholdRoadImage)

#for path in test_images_paths:

#    image  =  cv2.imread(path)
#    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

#    result = lineDetector.process_frame(image)
    
#    result = cv2.cvtColor(result,cv2.COLOR_RGB2BGR)
#    cv2.imshow("Hallo",result)
#    cv2.waitKey(0)
    
from moviepy.editor import VideoFileClip
video_output1 = 'challenge_video_output.mp4'
video_input1 = VideoFileClip('challenge_video.mp4')
processed_video = video_input1.fl_image(lineDetector.process_frame)
processed_video.write_videofile(video_output1, audio=False)