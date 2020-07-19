# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""


import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImagesOperations():
    """set of auxiliar functions to modify images"""

    def __init__(self):
        pass
    
    @staticmethod
    def image_resize(image, width = None, height = None, 
                                                    inter = cv2.INTER_AREA):
        """resized an image while keeping aspect ratio.
        """
        dim = None
        (h, w) = image.shape[:2]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation = inter)
        return resized,r

    @staticmethod
    def get_mask_roi(imgcv,use_body,dict_boxes,contrast):
        """get binary image of the region of interest roi. This process 
        determines how exact a measurement of the fish triat can be.
        """
        # compare the Body box against the distante
        # between Mouth and Eye to create a ROI
        # that has the fish in the center.
        boolean_roi=True
        boolean_Mou=True
        try:
            if use_body:
                #print("enter body")
                tl2=dict_boxes.get("Bod")[0]
                br2=dict_boxes.get("Bod")[1]
            else:
                #check if Mou and Tai were detected
                #print("else enter body")
                #print(dict_boxes.get("Mou"),dict_boxes.get("Tai"))
                if dict_boxes.get("Mou") and dict_boxes.get("Tai"):
                    tl2=dict_boxes.get("Mou")[0][0],dict_boxes.get("Tai")[0][1]
                    br2=dict_boxes.get("Tai")[1][0],dict_boxes.get("Tai")[1][1]
                    if 2*tl2[0]>=br2[0]:
                        boolean_Mou=False
                        tl2=0,dict_boxes.get("Tai")[0][1]
                else:
                    boolean_roi=False
        except:
            boolean_roi=False

        blank_image = np.zeros((imgcv.shape[0],imgcv.shape[1]), np.uint8)
        if boolean_roi:
            pixel_tolerance=20
            rate = lambda x: 0 if x<0 else x
            coor_1=rate(tl2[1]-pixel_tolerance)
            coor_2=rate(br2[1]+pixel_tolerance)
            coor_3=rate(tl2[0]-pixel_tolerance)
            coor_4=rate(br2[0]+pixel_tolerance)
            roi=imgcv[coor_1:coor_2,coor_3:coor_4]
        else:
            roi=imgcv
        alpha = contrast # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        #convertScaleAbs:
        #Scales, calculates absolute values, and converts the result to 8-bit.
        roi = cv2.convertScaleAbs(roi, alpha=alpha, beta=beta)


        # get the gray image of ROI
        gray = cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        #gray=cv2.equalizeHist(gray)

        # Apply a gaussian Convolution to smooth the
        # image 3x3 kernel.
        gray = cv2.GaussianBlur(gray, (5,5), 0)
    
        # Use threshold to generate a binary image.
        # here we Aply Otsu’s Binarization, since at this
        # point our image is bimodal image (In simple words, 
        # bimodal image is an image whose histogram has two peaks)
        thresh = cv2.threshold(gray, 0, 255, 
                            cv2.THRESH_OTSU)[1]
        #check if you need the invert the image.
        get_corner_kernel=np.mean(thresh[0:10,0:10])
        if get_corner_kernel>50:
            thresh=255-thresh

        # We need to ensure that the image is completely closed. 
        # use morphology to close all of the disconnected regions
        # together. The (2,2) can be thought as the minimum distance
        # between regions, that is considered to be joined.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                                kernel, iterations=10)

        # Find outer contour and fill with white
        # the insise of the closed contour.
        cnts = cv2.findContours(close,cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cv2.fillPoly(close, cnts, 1)

        if boolean_roi:
            blank_image[coor_1:coor_2,coor_3:coor_4]=close
            close=blank_image
        return boolean_Mou,close