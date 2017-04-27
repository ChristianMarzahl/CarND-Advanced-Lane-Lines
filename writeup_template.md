
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



I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

