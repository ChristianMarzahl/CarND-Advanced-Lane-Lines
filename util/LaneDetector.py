import numpy as np
import cv2

from util.CamCalibration import CameraCalibration
from util.PerspectiveTransformation import PerspectiveTransformation
from util.ThresholdRoadImage import ThresholdRoadImage
from util.Line import Line
from util.Line import LineType

class LaneDetector(object):
    """description of class"""

    def __init__(self, camera_calibration, perspective_transformation, threshold_road_image ):

        self.camera_calibration = camera_calibration
        self.perspective_transformation = perspective_transformation
        self.threshold_road_image = threshold_road_image

        self.left_line = Line(LineType.LeftLine)
        self.right_line = Line(LineType.RightLine)

        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = 30/720 # 30/720# 3.048/100 # meters per pixel in y dimension, lane line is 10 ft = 3.048 meters
        self.xm_per_pix = 3.7/700 # 3.7/660# 3.7/378 # meters per pixel in x dimension, lane width is 12 ft = 3.7 meters

        self.image_offset = 250
        self.dists = []
        # 
        self.line_segments = 9
        
    def process_frame(self, frame):
        
        input_frame = np.copy(frame)
        
        debug_dictionary = {"Input":input_frame}

        # undistort image 
        undist_image = self.camera_calibration.undistort_image(frame)
        debug_dictionary.update({"undistort_image":undist_image})

        # perspective transformation
        warped_image = self.perspective_transformation.apply(undist_image)
        debug_dictionary.update({"warped_image":warped_image})

        # threshold image 
        threshold_image = self.threshold_road_image.compute_threshold(warped_image)
        debug_dictionary.update({"threshold_image":threshold_image})

        self.left_line.update_line(threshold_image, debug_dictionary)
        self.right_line.update_line(threshold_image, debug_dictionary)

        # draw the current best fit if it exists
        if self.left_line.detected is not False and self.right_line.detected is not False:
            result =  self.draw_lane(input_frame,threshold_image,self.left_line,self.right_line)  

            left_curve_rad = self.left_line.calculate_curviness(threshold_image,self.ym_per_pix,self.xm_per_pix)
            right_curve_rad = self.right_line.calculate_curviness(threshold_image,self.ym_per_pix,self.xm_per_pix)
            curve_rad = np.average([left_curve_rad,right_curve_rad])

            center_dis = self.calculate_center_dist(threshold_image,self.left_line,self.right_line)

            result = self.draw_data(result,curve_rad, center_dis)
            debug_dictionary.update({"result":result})
            
            return self.generate_debug_view(debug_dictionary)
        
        return self.generate_debug_view(debug_dictionary)

    def generate_debug_view(self, debug_dictionary):

        mulit_image = np.zeros([1080, 1920,3],dtype=np.uint8)

        if 'result' in debug_dictionary:
            mulit_image[0:720,0:1280] = debug_dictionary['result']

        if 'result' not in debug_dictionary:
            mulit_image[0:720,0:1280] = debug_dictionary['Input']

        if 'warped_image' in debug_dictionary:
            resized = cv2.resize(debug_dictionary['warped_image'], (640,360), interpolation=cv2.INTER_AREA)
            mulit_image[0:360,1280:1280+640] = resized
    
        if 'threshold_image' in debug_dictionary:
            image = debug_dictionary['threshold_image'].astype(np.uint8)
            threshold_image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(threshold_image, (640,360), interpolation=cv2.INTER_AREA)
            mulit_image[360:360+360,1280:1280+640] = resized

        if 'result_polyfit_prev_LineType.LeftLine' in debug_dictionary:
            resized = cv2.resize(debug_dictionary['result_polyfit_prev_LineType.LeftLine'], (640,360), interpolation=cv2.INTER_AREA)
            mulit_image[720:720+360,0:640] = resized

        if 'result_polyfit_prev_LineType.RightLine' in debug_dictionary:
            resized = cv2.resize(debug_dictionary['result_polyfit_prev_LineType.RightLine'], (640,360), interpolation=cv2.INTER_AREA)
            mulit_image[720:720+360,640:640+640] = resized

        if 'result_sliding_window_LineType.LeftLine' in debug_dictionary:
            resized = cv2.resize(debug_dictionary['result_sliding_window_LineType.LeftLine'], (640,360), interpolation=cv2.INTER_AREA)
            mulit_image[720:720+360,0:640] = resized

        if 'result_sliding_window_LineType.RightLine' in debug_dictionary:
            resized = cv2.resize(debug_dictionary['result_sliding_window_LineType.RightLine'], (640,360), interpolation=cv2.INTER_AREA)
            mulit_image[720:720+360,640:640+640] = resized


        # histogram_
        h = np.zeros((256,1280,3))
        if 'histogram_LineType.RightLine' in debug_dictionary:
            try:
                hist = debug_dictionary['histogram_LineType.RightLine']
                cv2.normalize(hist,hist,0,255,cv2.NORM_MINMAX) 

                hist=np.int32(np.around(hist))
                hist = np.subtract(255,hist)
                bins = [i for i in range(0,1280)] # np.arange(256).reshape(256,1)
                pts = np.column_stack((bins,hist))

                cv2.polylines(h,[pts],False,(255,0,0), 5)
                resized = cv2.resize(h, (640,360), interpolation=cv2.INTER_AREA)
                mulit_image[720:720+360,1280:1280+640] = resized
            except:
                print ("Hist Error")

        mulit_image = cv2.resize(mulit_image, (1920,1080), interpolation=cv2.INTER_AREA)

        cv2.putText(mulit_image,"Birds eye",(1300,40), cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))

        cv2.putText(mulit_image,"Threshold",(1300,400), cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
        
        cv2.putText(mulit_image,"Histogram",(1300,760), cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))

        if 'result_sliding_window_LineType.RightLine' in debug_dictionary:
            cv2.putText(mulit_image,"Sliding Window",(600,760), cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))
        elif 'result_polyfit_prev_LineType.RightLine' in debug_dictionary:
            cv2.putText(mulit_image,"Polyfit",(600,760), cv2.FONT_HERSHEY_DUPLEX,1,(255,0,0))

        return mulit_image


    def draw_lane(self, original_img, binary_img, left_line, right_line):

        new_img = np.copy(original_img)
        if left_line.best_fit is None or right_line.best_fit is None:
            return original_img
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
        h,w = binary_img.shape
        ploty = np.linspace(0, h-1, num=h)# to cover same y-range as image
        left_fitx = left_line.best_fit[0]*ploty**2 + left_line.best_fit[1]*ploty + left_line.best_fit[2]
        right_fitx = right_line.best_fit[0]*ploty**2 + right_line.best_fit[1]*ploty + right_line.best_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
        cv2.polylines(color_warp, np.int32([pts_left]), isClosed=False, color=(255,0,255), thickness=15)
        cv2.polylines(color_warp, np.int32([pts_right]), isClosed=False, color=(0,255,255), thickness=15)

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = self.perspective_transformation.apply_inv(color_warp) 

        # Combine the result with the original image
        result = cv2.addWeighted(new_img, 1, newwarp, 0.5, 0)
        return result

    def calculate_center_dist(self, bin_img, left_line, right_line):
            
        center_dist = 0
        # Distance from center is image x midpoint - mean of l_fit and r_fit intercepts 
        if left_line.best_fit is not None and right_line.best_fit is not None:
            car_position = bin_img.shape[1]/2
            l_fit_x_int = left_line.best_fit[0]*bin_img.shape[0]**2 + left_line.best_fit[1]*bin_img.shape[0] + left_line.best_fit[2]
            r_fit_x_int = right_line.best_fit[0]*bin_img.shape[0]**2 + right_line.best_fit[1]*bin_img.shape[0] + right_line.best_fit[2]
            lane_center_position = (r_fit_x_int + l_fit_x_int) /2
            center_dist = (car_position - lane_center_position) * self.xm_per_pix
        return center_dist

    def draw_data(self, original_img, curv_rad, center_dist):
        output_image = np.copy(original_img)
        h = output_image.shape[0]
        font = cv2.FONT_HERSHEY_DUPLEX

        direction = ''
        if center_dist > 0:
            direction = ' right'
        elif center_dist < 0:
            direction = ' left'

        text = 'Radius: ' + '{:.0f}'.format(curv_rad) + 'm'
        cv2.putText(output_image, text, (40,70), font, 1, (255,0,0), 2, cv2.LINE_AA)

        text = 'Center distance: ' + '{:.0f} cm'.format(abs(center_dist) * 100) + direction
        cv2.putText(output_image, text, (750,70), font, 1, (255,0,0), 2, cv2.LINE_AA)

        return output_image