import numpy as np
import matplotlib.pyplot as plt
import cv2
import re




def get_low_value_thresh(img,gray=False,box=5):
    m,n=int(img.shape[0]/2),int(img.shape[1]/2)
    avg_center_picture=[]
    #box=5
    i,j=m-box,n-box
    while i<=m+box:
        while j<=n+box:
            avg_center_picture.append(img[i][j])
            #print(i,j)
            j+=1
        i+=1
    # multipled values found in the cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if(gray):
        minimun_intesity=int(np.mean(avg_center_picture))
    else:
        minimun_intesity=int(np.mean(np.mean(avg_center_picture,axis=0)*(0.299,0.587,0.114)))
    return minimun_intesity



def image_prepare(image,guasian_kernel_dim=(3,3),gray_step=True,blur_tec="gausian_smothing",resize=(640,360)):
    gray_image=image
    if(gray_step):
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
        minimun_intesity=get_low_value_thresh(gray_image,gray=gray_step)
        ret,thresh1 = cv2.threshold(gray_image,minimun_intesity,1,cv2.THRESH_BINARY)
        
    """
    if(blur_tec=="gausian_smothing"):
        #print('gausian_smothing')
        blur = cv2.GaussianBlur(gray_image, guasian_kernel_dim, 0)

    elif(blur_tec=="bilateral_smothing"):
        #print('bilateral_smothing')
        blur = cv2.bilateralFilter(src=gray_image,d=100,sigmaColor=75,sigmaSpace=75)
    """
    return thresh1
"""
def get_contour_mask(image_prepared):
    image_copy=image_prepared.copy()
    result=cv2.findContours(image_copy, 1, 2)
    contours= result[1]
    #print(contours[0])
    contours_mask = np.zeros( (image_copy.shape[0],image_copy.shape[1]) ) 
    cv2.fillPoly(contours_mask, pts =contours, color=(255,255,255))
    contours_mask=contours_mask/255
    #helper.display(contours_mask)
    return contours_mask
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

def multiple_image_contour(image,contours_mask):
    out = image.copy()
    out[:,:,0] =contours_mask*image[:,:,0]
    out[:,:,1] =contours_mask*image[:,:,1]
    out[:,:,2] =contours_mask*image[:,:,2]
    return out

"""
def tail_detection(contours_mask,dim_to_calculate,Tai_box,col_position_CPd=None):
    try:
        tail_code_vector=load_object('pickle_objects/tail_code_vector.file')

        detected_tail=template_matching(contours_mask,tail_code_vector,nearest_col=False,mode="remove",return_difference=True,tail=True)
        #display(detected_tail)

        if(dim_to_calculate=="CPd"):
            detected_tail=detected_tail[:,0:detected_tail.shape[1]-int((detected_tail.shape[1]*20)/100)]
        # detect last columns where the last no summation is
        sum_per_col=np.sum(detected_tail,axis=0)
        col_sum_index=[index for index,col in enumerate(sum_per_col) if col!=0]
"""

def gray_equalized(masked_image,greay_step=True):
    gray_image=masked_image
    if(greay_step):
        gray_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    eq_image=cv2.equalizeHist(gray_image)
    return eq_image