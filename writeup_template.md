
---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/Undist/calibration1.jpg "Original Image"
[image2]: ./output_images/Undist/undist.png "Undistorted Image"

[image3]: ./output_images/Pipeline/undist_image.png "Undistorted Image"
[image4]: ./output_images/Pipeline/warped_image.png "bird eye view"

[image5]: ./output_images/Pipeline/threshold_image.png "Threshold Image"

[image6]: ./output_images/Pipeline/result_sliding_window_LineType_left.png "Sliding Window Left"
[image7]: ./output_images/Pipeline/result_sliding_window_LineType_right.png "Sliding Window Right"
[image8]: ./output_images/Pipeline/histo.png "Histogram"

[image9]: ./output_images/Pipeline/result_polyfit_prev_LineType.LeftLine.png "Polyfit Left"
[image10]: ./output_images/Pipeline/result_polyfit_prev_LineType.RightLine.png "Polyfit Right"
[image11]: ./output_images/Pipeline/histo.png "Histogram"

[image12]: ./output_images/Pipeline/mulit_image.png "Debug View"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

### Camera Calibration

#### 1. Camera matrix and distortion coefficients

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

The code for the camera calibration is unter "util\CameraCalibration.py" contained in the class CameraCalibration. The class can load a ready calculated camera matrix and distortion coefficients from a pickle file or calculate and store them in one. The following code shniped shows the calculation.

```python
    def __calculate_calibrations_matrix(calibration_image_list, nx = 9, ny = 6):

        # Arrays to store object points and image points from all the images
        objpoints = [] 
        imgpoints = []
        
        # Prepare object points like (0,0,0), (1,0,0), (2,0,0) ..... ,(7,5,0)
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates

        img_size = None

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for fname in calibration_image_list:
            # read in each image in gray scale
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

            if img_size == None:
                img_size = (gray.shape[1], gray.shape[0])
    
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (nx,ny), corners,ret)
                cv2.imwrite(fname.replace(".jpg","_result.png"),img)
                #cv2.imshow('img',img)
                #cv2.waitKey(50)
            else:
                print("Not found: {0}".format(fname))

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)

        return mtx, dist
```

The following gif visualize the process of extracting the checkboard corners. 


The function undistort_image apply the undistortion matrix and distortion coefficient on the passed image.

```python
    def undistort_image(self,img):
        return cv2.undistort(img, self.camera_matrix, self.distortion_coefficients)
```

| Original Image        | Undistorted Image   | 
| :-------------:|:-------------:| 
| ![alt text][image1]      | ![alt text][image2] | 



### Pipeline (single images)

In the following steps I demonstrate the complete pipline perfomed on each input image. 

#### 1. Provide an example of a distortion-corrected image.

First for each image the function **undistort_image** from the class **CameraCalibration** is called witch uses the opencv function **cv2.undistort**

![alt text][image3]

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `apply(image)`.  The `apply()` function takes as inputs an image (`image`). The source (`transform_src`) and destination (`transform_dst`) points are saved as member variables of the class **PerspectiveTransformation** 

``` python

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

```
I chose to hardcode the source and destination points. This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570,460      | 450, 0        | 
| 710,460      | 830, 0      |
| 225,690     | 450, 720      |
| 1070,680      | 830, 720        |

I verified that my perspective transform was working as expected by visualizing the following brid eye view iamge.

![alt text][image4]


#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.


I used only two types of color thresholds to generate a binary image. The first threshold was performed in the HLS colorspace on the L channel with a range from 220 to 255. The second threshold was performed in the LAB colorspace on the L channel with a range from 190 to 255. Afterwards the two methods were combined to a combined_binary.

``` python

class ThresholdRoadImage(object):
    """description of class"""

    def __init__(self, sx_thresh = (20,100), s_thresh = (170,255)):

        self.sx_thresh = sx_thresh
        self.s_thresh = s_thresh

    def compute_threshold(self, img):
        
        #combined_binary = self.color_and_gradient(img)

        ## HLS Threshold
        img_HLS_Thresh = self.hls_thresh(img)

        ## Lab Threshold 
        img_LAB_Thresh = self.lab_thresh(img)
    
        ## Combine HLS and LAB channel thresholds
        combined_binary = np.zeros_like(img_LAB_Thresh)
        combined_binary[(img_HLS_Thresh == 1) | (img_LAB_Thresh == 1)] = 255

        return combined_binary

    def lab_thresh(self, img, thresh=(190,255), channel = 2): # 190
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
        lab_c = lab[:,:,channel]

        lab_c = lab_c * (255 / lab_c.max())

        binary_output = np.zeros_like(lab_c)
        binary_output[((lab_c > thresh[0]) & (lab_c <= thresh[1]))] = 1
        # 3) Return a binary image of threshold result
        return binary_output

    def hls_thresh(self, img, thresh=(220, 255), channel = 1):# 220
        # Convert to HLS color space
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        hls_c = hls[:,:,channel]

        hls_c = hls_c * (255 / hls_c.max())

        binary_output = np.zeros_like(hls_c)
        binary_output[(hls_c > thresh[0]) & (hls_c <= thresh[1])] = 1
        # Return a binary image of threshold result
        return binary_outputLAB
```

Here's an example of my output for this step.  

![alt text][image5]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The class **Line** with the methods **sliding_window_polyfit** and **polyfit_using_prev_fit** are used to identify the lane lines. 
The method **sliding_window_polyfit** uses the sliding window technique with eight windows to find regions with the most with pixels on the binary image. From the extracted pixel positions a polynominal fit second oder is calculated **fit = np.polyfit(y, x, 2)**. If in the previous image a lane could be found, the polynominal fit  is used to speed up the procces by calling the method **polyfit_using_prev_fit** which extract the pixel coordinates in the area of the previus fit. 


``` python

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

``` 

| Sliding Window Left        | Sliding Window Right   |  Histogramm   | 
| :-------------:|:-------------:| :-------------:|
| ![alt text][image6]      | ![alt text][image7] | ![alt text][image8]  |

``` python

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

``` 

| Polyfit Left        | Polyfit Right   |  Histogramm   | 
| :-------------:|:-------------:| :-------------:|
| ![alt text][image9]      | ![alt text][image10] | ![alt text][image11]  |


![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code for calculating the curviness is also in the class **Line** in the method **calculate_curviness**. The value of ym_per_pix is 30/720 which is meters per pixel in y dimension and for xm_per_pix is 3.7/700 which meters per pixel in x dimension. The mathematical detail are describted [hier](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). 

``` python
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
``` 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The method **draw_lane** from the class **LaneDetector** is used to plot the calculated information on the road.

``` python

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
```
The following image shows all information combined into one plot. 

![alt text][image12]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

1. The pipeline will faile due to bad light conditions which the threshold step can not handle. I spend a majority of my time experimenting with various methods to make the threshold process more robust but the results are not so good. To improve the threshold quality I tryed local and global histogram equalization methods and normalization techniques. 
2. One possible inprovment could be the use of a better fitting method like RANSAC.

