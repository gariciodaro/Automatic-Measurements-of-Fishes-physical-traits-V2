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


def main(path,path_save_image):
    """Execute on terminal. It shows the image, and it waits
    for user input:
        -v : Vertical axis flip
        -h : Horizontal axis flip
        -c : clockwise rotation
        -cc: counter clockwise rotation. 
    """
    transformations={'v':[cv2.flip,1],
                'h':[cv2.flip,0],
                'c':[cv2.rotate,cv2.ROTATE_90_CLOCKWISE],
                'cc':[cv2.rotate,cv2.ROTATE_90_COUNTERCLOCKWISE]}

    files=os.listdir(path)
    
    name_images=[file for file in files if 
                    (file.endswith('.jpg') or 
                    file.endswith('.png') or 
                    file.endswith('.JPG') or
                    file.endswith('.PNG'))]

    for index,name in enumerate(name_images):
        print('****************************')
        print(index)
        image = path+'/'+name
        print(name)
        imgcv_p = cv2.imread(image)
        #RGB_img = cv2.cvtColor(imgcv_p, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(10,7))
        plt.imshow(imgcv_p), plt.axis("on")
        plt.show()
        cv2.waitKey(0)
        plt.close()

        list_transform = input("transformations : ").split(" ")
        if list_transform[0]!="n":
            for transform in list_transform:
                imgcv_p=transformations.get(transform)[0](
                    imgcv_p,transformations.get(transform)[1])
        cv2.imwrite(path_save_image+"/"+name, imgcv_p) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", help="path to process")
    #parser.add_argument("--show_plot", help="show_plot. plot")
    args = parser.parse_args()
    main(args.path)