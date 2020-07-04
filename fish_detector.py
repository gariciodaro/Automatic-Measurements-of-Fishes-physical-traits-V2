


import os
import sys
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.segmentation import active_contour
# Absolute path of .current script
script_pos = os.path.dirname(os.path.abspath(__file__))
if not script_pos in sys.path:
    sys.path.append(script_pos)
from skimage.filters import gaussian
import numpy as np
import time
import cv2
from darkflow.net.build import TFNet
import pandas as pd

import tensorflow as tf

from Classes.ReferenceDetector import ReferenceDetector
from Classes.ImagesOperations import ImagesOperations 
from Classes.AuxiliarNameReference import AuxiliarNameReference
from Classes.SingleBoxAdjustment import SingleBoxAdjustment
from Classes.FishTraits import FishTraits
from Classes.TraitsDrawer import TraitsDrawer

import argparse

import warnings
warnings.filterwarnings('ignore')
options = {"pbLoad":"/media/gari/extra_ssd/folders/Jacobs_resourses/AMT_V2/built_graph/yolov2-custum-fish.pb",
            "metaLoad": "/media/gari/extra_ssd/folders/Jacobs_resourses/AMT_V2/built_graph/yolov2-custum-fish.meta",
            "gpu": 0.0,
            "threshold": 0.2 }
tfnet = TFNet(options)
plt.rcParams['image.cmap'] = 'gray'






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pictures", help="path to pictures. return CNN feature detection")
    parser.add_argument("--mode_ratio", help="in_name_TL, reference_tape")
    parser.add_argument("--path_template", help="if mode_ratio=reference_tape, pass the path to the template")
    parser.add_argument("--length", help="if mode_ratio=reference_tape, pass physical distante of tape")
    parser.add_argument("--show_detection", help="return CNN feature detection")
    parser.add_argument("--show_end_result", help="return image with dimension drawing")
    parser.add_argument("--debug", help="show inline prints")

    args = parser.parse_args()

    show_yolo_detection=args.show_detection
    show_end_result=args.show_end_result
    debug=args.debug
    mode_ratio=args.mode_ratio

    
    if show_yolo_detection=="True":
        show_yolo_detection=True
    else:
        show_yolo_detection=False
    if show_end_result=="True":
        show_end_result=True
    else:
        show_end_result=False

    if debug=="True":
        debug=True
    else:
        debug=False
    if debug: print(type(show_yolo_detection),show_end_result)

    if mode_ratio=="reference_tape":
        assert args.path_template
        assert args.length
        path_template=args.path_template
        template=cv2.imread(path_template)

    path=args.path_pictures
    images=[file for file in os.listdir(path) if 
            (file.endswith(".jpg") or file.endswith(".png") or 
                file.endswith(".JPG"))]

    for each_file in images:
        # Read image
        print(path+each_file)
        image_1=path+each_file
        imgcv_p = cv2.imread(image_1)

        imgcv,rr = ImagesOperations.image_resize(imgcv_p, height = 360)

        if mode_ratio=="reference_tape":
            rd=ReferenceDetector(template=template,length=int(args.length),
                linear_transform=rr)
            reference=rd.get_ratio(image=imgcv_p,show_on_picture=True)
            #print(reference)

        if mode_ratio=="in_name_TL":
            aux_ref=AuxiliarNameReference()
            reference=aux_ref.get_dim_from_name(each_file).get('TL_dim')
            #print(reference)

        # Use to CNN to parts detection
        results = tfnet.return_predict(imgcv)
        im_copy=imgcv.copy()
        # Get coordinates to calculate the ROI
        # Region of interest
        yolo_r=SingleBoxAdjustment()
        use_body,dict_boxes=yolo_r.yolo_refinement(
                                    results=results,
                                    copy_image=im_copy,
                                    show_yolo_detection=show_yolo_detection)

        boolean_Mou, closed_contour=ImagesOperations.get_mask_roi(imgcv,
                                                            use_body,
                                                            dict_boxes)
        #plt.imshow(closed_contour), plt.axis("on")
        #plt.show()
        fish_object=FishTraits(closed_contour=closed_contour, 
                                yolo_boxes=dict_boxes,
                                boolean_Mou=boolean_Mou,
                                mode_ratio=mode_ratio,
                                i_ref=reference)
        fish_object.get_pair_point_coordinates_traits()

        #print(fish_object.traits_distances)

        t_drawer=TraitsDrawer(image=im_copy,
                    dictionary_coor=fish_object.traits_coordinates,
                    dictionary_meas=fish_object.traits_distances)
        t_drawer.draw_measurements()

        final=t_drawer.image

        df_m=pd.DataFrame.from_dict(fish_object.traits_distances,
                                    orient='index',
                                    columns=["Measurement"]).T
        print(df_m)
        df_m

        copy_image=final.copy()

        RGB_img=copy_image
        if show_end_result:
            if debug: print("show_end_result",show_end_result)
            RGB_img = cv2.cvtColor(copy_image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(10,7))
            plt.imshow(RGB_img), plt.axis("on")
            plt.show()
        print("*"*100)

