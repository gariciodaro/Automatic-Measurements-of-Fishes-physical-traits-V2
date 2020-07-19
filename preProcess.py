# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""

import os
import re
import argparse
import cv2
import matplotlib.pyplot as plt


def main(imgcv_p,name,show_plot=False):
    """ Using the name of the file and its dimensions H,W.
    Transform it so that the mouth of the fish points left.
    """
    H,W,C=imgcv_p.shape
    if H>W:
        print("rotate")
        imgcv_p=cv2.rotate(imgcv_p, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if '.R.' in name:
        print('flip')
        imgcv_p = cv2.flip(imgcv_p, 1)
    if show_plot:
        RGB_img = cv2.cvtColor(imgcv_p, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,7))
        plt.imshow(RGB_img), plt.axis("on")
        plt.show()
        cv2.waitKey(0)
    return imgcv_p

