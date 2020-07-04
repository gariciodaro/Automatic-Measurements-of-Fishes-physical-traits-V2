import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
from Auxiliar.DictionaryAuxiliar import key_with_minval,key_with_maxval


def from_box_to_r_c(box):
    row0=box[0][1]
    row1=box[1][1]
    col0=box[0][0]
    col1=box[1][0]
    return [row0,row1,col0,col1]
"""
def auto_canny(image_1,image, sigma=0.20):
    #image_1 = cv2.GaussianBlur(image_1, (5,5), 0)
    #image = cv2.GaussianBlur(image, (5,5), 0)

    image = image.astype(np.uint8)
    image_1 = image_1.astype(np.uint8)
    # compute the median of the single channel pixel intensities
    v = np.median(image_1)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged
"""

def auto_canny(image, sigma=0.40):
    #image_1 = cv2.GaussianBlur(image_1, (5,5), 0)
    #image = cv2.GaussianBlur(image, (5,5), 0)

    image = image.astype(np.uint8)
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def tail_traits(close,dim_to_calculate,Tai_box):
    #plt.imshow(close), plt.axis("on")
    #plt.show()
    #transform rectangular tuples
    #to array in pixel space
    #to do the cropping
    if Tai_box:
        print(Tai_box)
        bc=from_box_to_r_c(Tai_box)

        #extra only the tail, while
        #preserving dimensions.
        get_tail_img=np.zeros(close.shape,np.uint8)
        get_tail_img[bc[0]:bc[1],bc[2]:bc[3]]=close[bc[0]:bc[1],bc[2]:bc[3]]
        #tail_canny=auto_canny(img,get_tail_img2)
        #plt.imshow(tail_canny), plt.axis("on")
        #plt.show()
        #plt.imshow(get_tail_img), plt.axis("on")
        #plt.show()
        if(dim_to_calculate=="CPd"):
            col_pos=bc[2]
            rear_col_pos=bc[2]

            row_values=get_tail_img[:,bc[2]]
            index_per_row_no_zero=[index for index,row in enumerate(row_values) if row!=0]

            #-6
            row_pos=index_per_row_no_zero[0]
            rear_row_pos=index_per_row_no_zero[-1]

        if(dim_to_calculate=="CFd"):
            col_pos=bc[3]
            rear_col_pos=bc[3]

            row_pos=bc[0]
            rear_row_pos=bc[1]

        #plt.imshow(close), plt.axis("on")
        #plt.show()
        
        #print(index_per_row_no_zero[0],index_per_row_no_zero[-1])

        #print(row_pos,col_pos,rear_row_pos,rear_col_pos)
        return [row_pos,col_pos,rear_row_pos,rear_col_pos]
    else:
        return None

def TL(Mou_box,Tai_box):
    if Mou_box and Tai_box:
        bm=from_box_to_r_c(Mou_box)
        bt=from_box_to_r_c(Tai_box)

        row_pos=int((bm[1]-bm[0])/2)+bm[0]
        rear_row_pos=row_pos

        col_pos=bm[2]
        rear_col_pos=bt[3]
        #print(row_pos,col_pos,rear_row_pos,rear_col_pos)
        return [row_pos,col_pos,rear_row_pos,rear_col_pos]
    else:
        return None

def eye_traits(gray,close,Eye_box,dim_to_calculate):
    if Eye_box:
        be=from_box_to_r_c(Eye_box)

        center_eye_row=int((be[1]-be[0])/2)+be[0]
        center_eye_col=int((be[3]-be[2])/2)+be[2]

        col_pos=rear_col_pos=center_eye_col

        if(dim_to_calculate=="Hd"):

            gray = cv2.cvtColor(gray,cv2.COLOR_BGR2GRAY)
            gray=cv2.equalizeHist(gray)
            gray = cv2.GaussianBlur(gray, (11,11), 0)
            gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

            """
            thresh = cv2.threshold(gray, 0, 255, 
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
            close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, 
                                    kernel, iterations=20)

            cnts = cv2.findContours(close,cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if len(cnts) == 2 else cnts[1]
            cv2.fillPoly(close, cnts, [255,255,255])
            

            plt.imshow(close*gray), plt.axis("on")
            plt.show()
            """
            canny_d = auto_canny(gray)
            #plt.imshow(canny_d), plt.axis("on")
            #plt.show()


            row_list=[index for index,each_row in 
                    enumerate(close[:,center_eye_col]) if each_row!=0]
            row_pos=row_list[0]
            rear_row_pos=center_eye_row

        if(dim_to_calculate=="Eh"):
            row_list=[index for index,each_row in 
                    enumerate(close[:,center_eye_col]) if int(each_row)==255]
            row_pos=center_eye_row
            rear_row_pos=row_list[-1]

        #print(row_pos,col_pos,rear_row_pos,rear_col_pos)
        return [row_pos,col_pos,rear_row_pos,rear_col_pos]
    else:
        return None






def dim_plotter(image,two_pairs_coordinates,color,show_imaga=True):
    
    image_copy=image.copy()
    p1_row, p1_col =two_pairs_coordinates[0],two_pairs_coordinates[1]
    p2_row, p2_col =two_pairs_coordinates[2],two_pairs_coordinates[3]
    lineThickness = 1
    cv2.line(image_copy, (p1_col,p1_row), (p2_col,p2_row), color, lineThickness)

    cv2.circle(img=image_copy, center=(p1_col,p1_row),
                  radius=6, color=color,
                  thickness=1, lineType=10, shift=0)
    cv2.circle(img=image_copy, center=(p2_col,p2_row),
                  radius=6, color=color,
                  thickness=1, lineType=10, shift=0)

    #cv2.line(image_copy, (p1_col,p1_row), (p2_col,p2_row-10), (255,0,0), lineThickness)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #leng_line=int(col_pos_tail/2)
    #cv2.circle(img=image_copy, center=(leng_line+10,horital_pos_line),
    #              radius=12, color=(255,255,255),
    #              thickness=-1, lineType=10, shift=0)
    #cv2.putText(image_copy,dim_text,(leng_line,horital_pos_line+6), font, 0.4,(255,0,0),1,cv2.LINE_AA)
    
    
    #cv2.line(image_copy, (col_pos, 0), (col_pos, row_pos), (255,0,0), lineThickness)
    #cv2.line(image_copy, (col_pos_tail, 0), (col_pos_tail, row_pos_tail), (255,0,0), lineThickness)

    if(show_imaga):
        plt.imshow(image_copy), plt.axis("on")
        plt.show()
    else:
        return image_copy