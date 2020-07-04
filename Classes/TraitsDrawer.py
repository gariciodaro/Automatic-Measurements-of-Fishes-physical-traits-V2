# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np


class TraitsDrawer():
    """Drawer over canvas image the detected measured traits"""
    def __init__(self, image,dictionary_coor,dictionary_meas):
        self.image=image
        self.dictionary_coor=dictionary_coor
        self.dictionary_meas=dictionary_meas


    def draw_measurements(self):
        colors=[(255,0,122),(0,0,114),(233,0,0),
        (240,128,0),(205,0,92),(255,255,255),(178,34,34),
        (0,0,0),(0,0,0),(0,0,128),(0,255,0),(0,0,255),
        (255,0,0)]
        i=0
        for key,each_measure in self.dictionary_coor.items():
            if each_measure:
                self.image=self.dim_plotter(self.image,
                                    each_measure,
                                    colors[i],
                                    self.dictionary_meas.get(key),
                                    False)
            i+=1


    @staticmethod
    def dim_plotter(image,two_pairs_coordinates,color,value,show_imaga=True):
        image_copy=image.copy()
        p1_row, p1_col =two_pairs_coordinates[0],two_pairs_coordinates[1]
        p2_row, p2_col =two_pairs_coordinates[2],two_pairs_coordinates[3]

        lineThickness = 1
        cv2.line(image_copy, (p1_col,p1_row), (p2_col,p2_row), color,
                                                                lineThickness)

        cv2.circle(img=image_copy, center=(p1_col,p1_row),
                      radius=6, color=color,
                      thickness=1, lineType=10, shift=0)
        cv2.circle(img=image_copy, center=(p2_col,p2_row),
                      radius=6, color=color,
                      thickness=1, lineType=10, shift=0)

        if(show_imaga):
            plt.imshow(image_copy), plt.axis("on")
            plt.show()
        else:
            return image_copy
