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
    def get_dim_from_name(string):
        """ take the string remove everything
        but numbers and decimal points.

        Returns
        -------
        new_numbers_only : list
            decimal entries of the input.
        """
        dic_dim={}
        dic_dim['TL_dim']=1
        string=str.strip(string.lower())
        string=string.replace(".jpg", 
                                "").replace(".png", "").replace(".jpeg", "")
        if '.tl.' in string:
            string=string.split('.tl.')[0]
            NUMBER_ONLY = re.compile('[^0-9.]')
            numbers_only = str.strip(NUMBER_ONLY.sub(" ",string)).split()
            numbers_only= [ each for each in numbers_only if each!= "."]
            fix_dot_init= lambda x : "".join(x[1:]) if  x[0]=="."  else x
            fix_do_last= lambda x : "".join(x[0:-1]) if  x[-1]=="."  else x
            numbers_only=[ fix_dot_init(each) for each in numbers_only ]
            numbers_only=[ fix_do_last(each) for each in numbers_only ]
            try :
                dic_dim['TL_dim']=float(numbers_only[-1])
            except:
                pass
        print(string)
        print(dic_dim)
        return dic_dim
