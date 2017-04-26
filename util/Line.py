import numpy as np
import cv2
from scipy import signal
from enum import Enum

class LineType(Enum):
    RightLine = 1
    LeftLine = 2

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self, line_type):

        self.line_type = line_type

        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        self.best_fit_poly = None
        #polynomial coefficients for the most recent fit
        self.current_fit = []
        self.current_fit_poly = None
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        # number of sliding windows
        self.sliding_windows = 8        
        # Set the width of the windows +/- margin
        self.margin = 100

        # check if there is a better var
        self.lane_inds = None

    def update_line(self, image, debug_dictionary):

        # if on the last image was not a line found use sliding window 
        if not self.detected:
            fit, self.lane_inds = self.sliding_window_polyfit(image, debug_dictionary)
        else:
            fit, self.lane_inds = self.polyfit_using_prev_fit(image, self.best_fit, debug_dictionary)

            # check fit, if the fit is not good retry with sliding window 
            if fit is not None and self.best_fit is not None:
                percent_diff = (np.sum(np.abs(1-(self.best_fit/fit))))
                if (percent_diff > 1 and len(self.current_fit) > 0):
                    fit, self.lane_inds = self.sliding_window_polyfit(image, debug_dictionary)
                    self.best_fit = None                     


        # check if new fit is plossible 
        if fit is not None:
            percent_diff = 0
            if self.best_fit is not None:
                # if we have a best fit, see how this new fit compares
                self.diffs = abs(fit-self.best_fit)
                percent_diff = (np.sum(np.abs(1-(self.best_fit/fit))))
            if (percent_diff > 1 and len(self.current_fit) > 0):
                self.detected = False
                print (self.detected)
            else:
                self.detected = True
                self.current_fit.append(fit)
                if len(self.current_fit) > 5:
                    # throw out old fits, keep newest n
                    self.current_fit = self.current_fit[len(self.current_fit)-5:]
                self.best_fit = np.average(self.current_fit, axis=0)
        # or remove one from the history, if not found
        else:
            self.detected = False
            if len(self.current_fit) > 0:
                # throw out oldest fit
                self.current_fit = self.current_fit[:len(self.current_fit)-1]
            if len(self.current_fit) > 0:
                # if there are still any fits in the queue, best_fit is their average
                self.best_fit = np.average(self.current_fit, axis=0)


    def sliding_window_polyfit(self, img, debug_dictionary):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]//2)
        quarter_point = np.int(midpoint//2)

        base = None
        if self.line_type == LineType.LeftLine:
            base = np.argmax(histogram[quarter_point:midpoint]) + quarter_point
        else:
            base = np.argmax(histogram[midpoint:(midpoint+quarter_point)]) + midpoint

        # Set height of windows
        window_height = np.int(img.shape[0]/self.sliding_windows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        x_current = base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []
        # Rectangle data for queued for debug reasons
        rectangle_data = []

        # Step through the windows one by one
        for window in range(self.sliding_windows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = img.shape[0] - (window+1)*window_height
            win_y_high = img.shape[0] - window*window_height
            win_x_low = x_current - self.margin
            win_x_high = x_current + self.margin
            rectangle_data.append((win_y_low, win_y_high, win_x_low, win_x_high))
            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) & (nonzerox < win_x_high)).nonzero()[0]
            # Append these indices to the lists
            lane_inds.append(good_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 

        fit = None

        if len(x) != 0:
            fit = np.polyfit(y, x, 2)

            # save the histogram and sliding window results for debug 
            result = self.draw_visualization_sliding_window(fit, lane_inds,img,rectangle_data)
            debug_dictionary.update({"result_sliding_window_{}".format(self.line_type):result})
        
        debug_dictionary.update({"histogram_{}".format(self.line_type):histogram})
        return fit, lane_inds

    def polyfit_using_prev_fit(self, img, fit_prev, debug_dictionary):

        # Take a histogram of the bottom half of the image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)

        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        lane_inds = ((nonzerox > (fit_prev[0]*(nonzeroy**2) + fit_prev[1]*nonzeroy + fit_prev[2] - self.margin)) & 
                      (nonzerox < (fit_prev[0]*(nonzeroy**2) + fit_prev[1]*nonzeroy + fit_prev[2] + self.margin))) 

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 

        fit_new = None
        if len(x) != 0:
            fit_new = np.polyfit(y, x, 2)

            # save the visualisation results for debug 
            result = self.draw_visualization_prev_fit(img,fit_new,lane_inds)
            debug_dictionary.update({"result_polyfit_prev_{}".format(self.line_type):result})

        debug_dictionary.update({"histogram_{}".format(self.line_type):histogram})
        return fit_new, lane_inds
    
    def draw_visualization_sliding_window(self, fit, lane_inds, image, rectangle_data):

        h = image.shape[0]
        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
     

        # Create an output image to draw on and  visualize the result
        out_img = np.uint8(np.dstack((image, image, image))*255)
        # Draw the windows on the visualization image
        for rect in rectangle_data:
            cv2.rectangle(out_img,(rect[2],rect[0]),(rect[3],rect[1]),(0,255,0), 2)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if self.line_type == LineType.LeftLine:
            out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
        else:
            out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [0, 0, 255]

        return out_img

    def draw_visualization_prev_fit(self, image, fit, lane_inds):
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((image, image, image))*255
        window_img = np.zeros_like(out_img)
        
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Color in left and right line pixels
        if self.line_type == LineType.LeftLine:        
            out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [255, 0, 0]
        else:
            out_img[nonzeroy[lane_inds], nonzerox[lane_inds]] = [0, 0, 255]

        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        line_window1 = np.array([np.transpose(np.vstack([fitx-self.margin, ploty]))])
        line_window2 = np.array([np.flipud(np.transpose(np.vstack([fitx+self.margin, ploty])))])
        line_pts = np.hstack((line_window1, line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        return result

    def calculate_curviness(self, image, ym_per_pix, xm_per_pix):
        curve_rad = 0

        # Define y-value where we want radius of curvature
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
        y_eval = np.max(ploty)

        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Extract line pixel positions
        x = nonzerox[self.lane_inds]
        y = nonzeroy[self.lane_inds] 

        if len(x) != 0:
            # Fit new polynomials to x,y in world space
            fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
            # Calculate the new radii of curvature
            curve_rad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

        return curve_rad;

