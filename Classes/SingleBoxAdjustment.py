# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""

from Classes.DictionaryAuxiliar import key_with_maxval
import numpy as np
import cv2
import matplotlib.pyplot as plt


class SingleBoxAdjustment():
    """Single Box Adjustment allow to refined the yolo detection boundary
    boxes. Only the hightest prediction per fish feature is allowed.
    """
    def __init__(self):
        pass

    def yolo_refinement(self,results,copy_image,
                                            show_yolo_detection,debug=False):
        """
        Takes yolo resutls a get's the maximum probability box

        results : dict
            yolo boxes detection.
        copy_image : numpy.ndarray
            a copy of the image in case drawing is required
        show_yolo_detection : boolean
        debug : boolean
        """
        self.results = results
        self.copy_image = copy_image.copy()
        self.show_yolo_detection = show_yolo_detection
        self.debug = debug
        colors=[tuple(255 * np.random.rand(3)) for i in range(5)]
        Boca_list={}
        Ojo_list={}
        aleta_list={}
        cuerpo_list={}
        cola_list={}
        key_Boca,key_Ojo,key_cuerpo,key_aleta,key_cola=9999,9999,9999,9999,9999
        
        for index,each_result in enumerate(results):
            label_in=each_result.get("label")
            
            if(label_in=="Boca"):
                Boca_list[index]=each_result.get("confidence")

            if(label_in=="Ojo"):
                Ojo_list[index]=each_result.get("confidence")

            if(label_in=="cuerpo"):
                cuerpo_list[index]=each_result.get("confidence")

            if(label_in=="aleta"):
                aleta_list[index]=each_result.get("confidence")

            if(label_in=="cola"):
                cola_list[index]=each_result.get("confidence")

        key_Boca=key_with_maxval(Boca_list)
        key_Ojo=key_with_maxval(Ojo_list)
        key_cuerpo=key_with_maxval(cuerpo_list)
        key_aleta=key_with_maxval(aleta_list)
        key_cola=key_with_maxval(cola_list)

        highest_key=[key_Boca,key_Ojo,key_cuerpo,key_aleta,key_cola]
        labels_name={"Boca":"Mou","Ojo":"Eye","cuerpo":"Bod",
                                                    "aleta":"Fin","cola":"Tai"}

        use_body=False
        Mou_box=Eye_box=Bod_box=Fin_box=Tai_box=0,0,False,"label"
        i=1
        for color , result in zip(colors, highest_key):
            if result!=9999:
                tl = (results[result]['topleft']['x'], 
                                            results[result]['topleft']['y'])
                br = (results[result]['bottomright']['x'],
                                        results[result]['bottomright']['y'])
                label = labels_name.get(results[result]['label'])
                if(label=="Mou"):
                    Mou_box=tl,br,True,label

                if(label=="Eye"):
                    Eye_box=tl,br,True,label

                if(label=="Bod"):
                    if debug: print(results[result].get("confidence"))
                    body_conf=results[result].get("confidence")
                    if(body_conf>0.1):
                        use_body=True
                        Bod_box=tl,br,True,label
                if(label=="Fin"):
                    Fin_box=tl,br,True,label

                if(label=="Tai"):
                    Tai_box=tl,br,True,label

                if show_yolo_detection:
                    if debug: print("Tf show_yolo_detection",
                                                    show_yolo_detection)
                    cv2.rectangle(self.copy_image , tl, br, color, 1)
                    cv2.putText(self.copy_image , label, tl, 
                    cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
            if show_yolo_detection and i==len(highest_key):
                RGB_img = cv2.cvtColor(self.copy_image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(10,7))
                plt.imshow(RGB_img)
                plt.show()
            i+=1
        boxes=[Mou_box,Eye_box,Bod_box,Fin_box,Tai_box]
        dict_boxes={each_box[-1]:each_box[0:3] for each_box in boxes if 
                                                            each_box[2]==True}
        return use_body,dict_boxes