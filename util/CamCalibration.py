import numpy as np
import cv2
import pickle

class CameraCalibration(object):
    """description of class"""

    def __init__(self, pickle_file="camera_calibration.pickle"):

        self.__load_pickle(pickle_file)

    @classmethod
    def calculate_and_save_calibrations_matrix(self, calibration_image_list, nx, ny,   output_pickle = "camera_calibration.pickle"):

        camera_matrix, distortion_coefficients = self.__calculate_calibrations_matrix(calibration_image_list, nx, ny)

        self.__save_calibrations_matrix(camera_matrix, distortion_coefficients,  output_pickle)


    def __load_pickle(self, file="camera_calibration.pickle"):
        with open(file, "rb") as f:
            matrix_data = pickle.load(f)
        self.camera_matrix = matrix_data['camera_matrix']
        self.distortion_coefficients = matrix_data['distortion_coefficients']

    def __save_calibrations_matrix(camera_matrix , distortion_coefficients , file="camera_calibration.pickle"):
        with open(file, "wb") as f:
            pickle.dump({"camera_matrix":camera_matrix,"distortion_coefficients":distortion_coefficients}, f)

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

    def undistort_image(self,img):
        return cv2.undistort(img, self.camera_matrix, self.distortion_coefficients)


        
         
