import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
from Auxiliar.DictionaryAuxiliar import key_with_maxval

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized





def calculate_angle(coor1,coor2,coor3):
    # coor1 boca
    # coor2 cola top
    # coor3 cola boton

    top_boca_x=coor1[0]
    top_boca_y=coor1[1]

    top_cola_x=coor2[0]
    top_cola_y=coor2[1]

    botom_cola_y=coor3[1]

    print("top_boca_x",top_boca_x)
    print("top_boca_y",top_boca_y)
    print("top_cola_x",top_cola_x)
    print("top_cola_y",top_cola_y)
    print("botom_cola_y",botom_cola_y)

    med=abs(botom_cola_y)/2



    delta_x=abs(top_boca_x-top_cola_x)

    delta_y=top_boca_y-med

    theta=np.arctan(delta_y/delta_x)

    print("med",med)
    print("delta_x",delta_x)
    print("delta_y",delta_y)

    return theta


def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

colors=[tuple(255 * np.random.rand(3)) for i in range(5)]


def draw_searh_cirle(image_shape):
    #row_pos,col_pos          =extreme_points[0],0
    #row_pos_tail,col_pos_tail=extreme_points[2],image_shape[1]
   
    center_image_x=image_shape[1]/2
    center_image_y=image_shape[0]/2
   
    s = np.linspace(0, 2*np.pi, 400)
    x = center_image_x  + (center_image_x)*np.cos(s)
    y = center_image_y  + (center_image_y)*np.sin(s)
    init = np.array([x, y]).T
    return init


def tensor_flow_op(results,copy_image,show_yolo_detection,debug=False):
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
    labels_name={"Boca":"Mou","Ojo":"Eye","cuerpo":"Bod","aleta":"Fin","cola":"Tai"}
    left=(0,0)
    right=(0,0)
    big_top_col,big_top_row=0,0

    #def calculate_angle(coor1,coor2,coor3):
    # coor1 boca
    # coor2 cola top
    # coor3 cola boton
    coor1=[0.1,0.1]
    coor2=[0.1,0.1]
    coor3=[0.1,0.1]
    use_body=False
    Mou_box=Eye_box=Bod_box=Fin_box=Tai_box=0,0,False,"label"
    for color , result in zip(colors, highest_key):
        if result!=9999:
            
            tl = (results[result]['topleft']['x'], results[result]['topleft']['y'])
            br = (results[result]['bottomright']['x'], results[result]['bottomright']['y'])
            label = labels_name.get(results[result]['label'])
            if(label=="Mou"):
                big_top_col=tl[0]
                coor1=[tl[0],tl[1]]
                Mou_box=tl,br,True,label

            if(label=="Eye"):
                Eye_box=tl,br,True,label

            if(label=="Bod"):
                if debug: print(results[result].get("confidence"))
                body_conf=results[result].get("confidence")
                if(body_conf>0.5):
                    use_body=True
                    Bod_box=tl,br,True,label

            if(label=="Fin"):
                Fin_box=tl,br,True,label

            if(label=="Tai"):
                right=br
                big_top_row=tl[1]
                coor2=[tl[0],tl[1]]
                coor3=[br[0],br[1]]
                Tai_box=tl,br,True,label

            if show_yolo_detection:
                if debug: print("Tf show_yolo_detection",show_yolo_detection)
                cv2.rectangle(copy_image, tl, br, color, 1)
                #cv2.imshow('secondCropImg', copy_image)
                cv2.putText(copy_image, label, tl, 
                cv2.FONT_HERSHEY_TRIPLEX, 1, color, 1)
                plt.imshow(copy_image)
                #plt.axis("off")
                plt.show()
    boxes=[Mou_box,Eye_box,Bod_box,Fin_box,Tai_box]

    return coor1,coor2,coor3,right,big_top_col,big_top_row,use_body,boxes

"""
def get_contour_mask(image_prepared):
    #save_object("./image_prepared",image_prepared)
    image_copy=image_prepared.copy()

    #result=cv2.findContours(image_copy, 1, 2)
    coun,herar=cv2.findContours(image_copy, mode=cv2.RETR_TREE , method=cv2.CHAIN_APPROX_SIMPLE, offset=(0, 0))
    maxx=[len(z) for z in coun]
    max_index=maxx.index(max(maxx))

    contours= coun[max_index]
    #print(contours[0])
    contours_mask = np.zeros( (image_copy.shape[0],image_copy.shape[1]) ) 
    #print(type(contours))
    #print(type(contours_mask))
    cv2.fillPoly(contours_mask, pts =contours, color=255)
    contours_mask=contours_mask/255
    #helper.display(contours_mask)
    return contours_mask


def auto_canny(image, sigma=0.20):
    image = image.astype(np.uint8)
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
"""
