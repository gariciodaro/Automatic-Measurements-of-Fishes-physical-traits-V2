# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""
import re
import numpy as np


class AuxiliarNameReference(object):
    """set of function to extract TL from name of the file"""
    def __init__(self):
        pass

    @staticmethod
    def check_name_for_SL_TL(string):
        string=string.lower()
        for_sl=string.split('sl')
        for_tl=string.split('tl')
        try:
            if len(for_sl)>1 and len(for_tl)>1:
                checker=[True,True]
            elif len(for_sl)>1 and len(for_tl)==1:
                checker=[True,False]
            elif len(for_sl)==1 and len(for_tl)>1:
                checker=[False,True]
            else:
                checker=[False,False]
        except:
            checker=[False,False]
        return checker

    @staticmethod
    def get_dim_from_name(string):
        string=str.strip(string.lower())
        string=string.replace(".jpg", 
                                "").replace(".png", "").replace(".jpeg", "")
        checker=AuxiliarNameReference.check_name_for_SL_TL(string)
        NUMBER_ONLY = re.compile('[^0-9.]')
        numbers_only = str.strip(NUMBER_ONLY.sub(" ",string)).split()
        SL_dim,TL_dim=0,0
        dic_dim={}
        if(checker[1]):
            dic_dim['TL_dim']=np.max(np.array(numbers_only).astype(float))
        elif(checker[0]==True and checker[1]==False):
            dic_dim['SL_dim']=np.max(np.array(numbers_only).astype(float))
        else:
            dic_dim['TL_dim']=0
        return dic_dim