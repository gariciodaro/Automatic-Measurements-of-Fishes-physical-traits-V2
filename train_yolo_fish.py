import matplotlib.pyplot as plt
import numpy as np

from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolov2-custum-fish.cfg", 
           "load": "bin/yolov2.weights",
           "batch": 5,
           "epoch": 100,
           "gpu": 0.8,
           "train": True,
           "annotation": "./fish_training/Annotations_fish/",
           "dataset": "./fish_training/Images_fish/"}

tfnet = TFNet(options)


tfnet.train()


tfnet.savepb()