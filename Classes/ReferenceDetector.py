# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""

#linear algebra
import numpy as np

#conda install opencv
import cv2
import os
import sys



class ReferenceDetector():
    """
    Class to obtain the ratio pixel/length in the picture

    Methods
    -------
    calculate_roi(image):
        calculates from passed image the region of interes ROI. The corner
        of the reference tape in the real image should end up in the in the
        near the 0,0 coordinate of the ROI.
    get_ratio(image,show_on_picture=False)
        calculates the ratio length(cm,mm... etc)/pixels. Uses calculate_roi
    """
    def __init__(self, template,length,linear_transform):
        """
        Parameters
        ----------
        template : numpy.ndarray
            image template of reference tape. The reference tape of all your
            images should be similar, if not, you must passed different 
            templates
        lenght : int
            physical lenght of the tape. e.g. 5. meaning 5cm
        """
        self.length = length
        self.template = template
        self.linear_transform = linear_transform

    def calculate_roi(self,image):
        """
        calculates from passed image the region of interes ROI. The corner
        of the reference tape in the real image should end up in the in the
        near the 0,0 coordinate of the ROI.

        Parameters
        ----------
        image : numpy.ndarray
            image should have 3 channels.

        Returns
        -------
        roi : numpy.ndarray
            rectangular area of interest ROI. This image is in gray scale,
            as a result it only has 1 channel.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        gray_template = cv2.cvtColor(self.template,
                            cv2.COLOR_BGR2GRAY).astype(np.uint8)
        #detect using the template
        res = cv2.matchTemplate(gray,gray_template,cv2.TM_CCOEFF_NORMED)
        threshold = 0.99
        loc = np.where( res >= threshold)
        while len(loc[0])==0:
            threshold=threshold-0.01
            loc = np.where( res >= threshold)
        #assign coordinates
        min_row=min(loc[0])
        max_row=min(loc[0])+gray_template.shape[0]

        min_col=min(loc[1])
        max_col=max(loc[1])+gray_template.shape[1]
        #crop gray image
        roi=gray[min_row: , min_col:]
        roi_color=image[min_row: , min_col:]
        return roi,roi_color

    def get_ratio(self,image,show_on_picture=False):
        """
        Calculates the ratio length(cm,mm... etc)/pixels. Uses calculate_roi.

        Parameters
        ----------
        image : numpy.ndarray
            image should have 3 channels.
        show_on_picture : boolean
            whether to show an image with the line crossing the detected
            reference tape.

        Returns
        -------
        ratio_length_pixel : int
        """
        ratio_length_pixel=None
        if show_on_picture:
            #plotting library for python 
            import matplotlib.pyplot as plt 
            #set black and white plotting
            plt.rcParams['image.cmap'] = 'gray'
        image_rectangular_roi,roi_color=self.calculate_roi(image)
        thresh = cv2.threshold(image_rectangular_roi, 0, 255,
                cv2.THRESH_OTSU)[1]
        #check if you need the invert the image.
        thresh_dim_col=int(thresh.shape[1]*50/100)
        get_corner_kernel=np.mean(thresh[0,thresh_dim_col-10:thresh_dim_col])
        if get_corner_kernel>10:
            thresh=255-thresh
        d_max=0
        #plt.imshow(thresh), plt.axis("on")
        #plt.show()
        image_copy=roi_color.copy()
        look_row=int(thresh.shape[0]*15/100)
        look_col=int(thresh.shape[1]*50/100)
        coor_line=[]
        for row in range(look_row):
            list_col=[]
            last_index=0
            for index,col in enumerate(thresh[row,0:look_col]):
                if col!=0:
                    list_col.append(index)
                    last_index=index
                if index-last_index>100 and len(list_col)!=0:
                    draw=False
                    break
            if len(list_col)!=0:
                d=list_col[-1]-list_col[0]
                if d_max<d:
                    d_max = d
                    p1_col=list_col[0]
                    p1_row=row
                    p2_col=list_col[-1]
                    p2_row=row
                    coor_line.append([p1_col,p1_row, p2_col,p2_row])
        if show_on_picture:
            print("maximum pixel distance",d_max)
            cv2.line(image_copy,
                (coor_line[-1][0],coor_line[-1][1]),
                (coor_line[-1][2],coor_line[-1][3]),
                (255,0,0),
                10)
            plt.imshow(image_copy), plt.axis("on")
            plt.show()
        if d_max != 0:
            d_max=d_max*self.linear_transform
            ratio_length_pixel=self.length/d_max
        return ratio_length_pixel