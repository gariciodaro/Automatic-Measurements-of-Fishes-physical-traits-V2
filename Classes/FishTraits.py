# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 2019

@author: gari.ciodaro.guerra
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import stats

class FishTraits(object):
    """Class to measure fish's traits."""

    def __init__(self, closed_contour, yolo_boxes,boolean_Mou,mode_ratio,i_ref):
        """Constructor

        Parameters
        ----------
        closed_contour : numpy.ndarray
            binary image of closed contour of the fish.
        yolo_boxes : dictionary
            contains the coordinates of the boxes detected
            by yolo:.
                keys(Mou). array cartesian coordinates of the box where 
                the mouth is located. ((x1,y1),(x2,y2)).Left superior 
                edge, right inferior edge in pixel coordinate system.
                (Tai). array cartesian coordinates of the box where the tail
                is located.
                (Eye). array cartesian coordinates of the box where the eye.
                (Fin). location of the fin.
                (Bod). location of the body.
        boolean_Mou : boolean
            whether the mouth location can be trusted. In 
            case False, the position of the yolo_boxes.get("Eye") is used 
            instead, for example for estimating maximal distances.
        mode_ratio : str
            can be in_name_TL or reference_tape.
        i_ref : float
            if mode_ratio=="in_name_TL" i_ref is TL in units of distance
            extracted from the name of the fila of the original fish. 
            if mode_ratio=="reference_tape" i_ref is directly the 
            pixels/[units of distance.]
        """
        self.closed_contour = closed_contour
        self.yolo_boxes = yolo_boxes
        self.boolean_Mou=boolean_Mou
        self.mode_ratio=mode_ratio
        self.i_ref=i_ref
        self.CPd_pp=self.CFd_pp=self.TL_pp=self.Mo_pp=self.Bl_pp=None
        self.PFi_pp=self.PFb_pp=self.PFl_pp=self.Hd_pp=self.Eh_pp=None
        self.Ed_pp=None

    def get_pair_point_coordinates_traits(self):
        """main function to calculate all the available features. Notice that
        some methods have dependency between them, so this function shows the
        recommended order of executions. 
        """

        self.tail_traits(dim_to_calculate="CPd")
        self.tail_traits(dim_to_calculate="CFd")

        # In case CPd did not have a proper edge
        # modify the trait to half CFd. Simitric.
        try:
            front_tail_row=self.CPd_pp[0]
            rear_tail_row=self.CFd_pp[0]
            if front_tail_row==rear_tail_row:
                self.CPd_pp[0]=\
                    self.CFd_pp[0]+int((self.CFd_pp[2]-self.CFd_pp[0])/4)
                self.CPd_pp[2]=\
                    self.CPd_pp[2]-int((self.CFd_pp[2]-self.CFd_pp[0])/4)
        except:
            pass

        self.TL()
        self.eye_traits(dim_to_calculate="Hd")
        self.eye_traits(dim_to_calculate="Eh")
        self.eye_traits(dim_to_calculate="Ed")
        self.Mo()
        self.Bl()
        if self.boolean_Mou:
            try:
                self.Eh_pp[2]=self.Mo_pp[2]
            except:
                pass
        self.fin_traits(dim_to_calculate="PFi")
        self.fin_traits(dim_to_calculate="PFb")
        self.fin_traits(dim_to_calculate="PFl")

        # if PFb_pp exist, it should correspont
        # to the max vertical lenght Bd
        if self.PFb_pp:
            self.Bd_pp=self.PFb_pp
        else:
            self.Bd()

        traits={"CPd":self.CPd_pp,
                "CFd":self.CFd_pp,
                "TL":self.TL_pp,
                "Mo":self.Mo_pp,
                "Bl":self.Bl_pp,
                "PFi":self.PFi_pp,
                "PFb":self.PFb_pp,
                "PFl":self.PFl_pp,
                "Hd":self.Hd_pp,
                "Eh":self.Eh_pp,
                "Ed":self.Ed_pp,
                "Bd":self.Bd_pp}
        self.traits_coordinates=traits

        if self.mode_ratio=="in_name_TL":
            TL_coordinates=self.traits_coordinates.get("TL")
            if TL_coordinates and self.i_ref:
                tl_last_coor=np.array((TL_coordinates[0],TL_coordinates[1]))
                tl_start_coor=np.array((TL_coordinates[2],TL_coordinates[3]))
                distance_px_tl=np.linalg.norm(tl_last_coor-tl_start_coor)
                self.ratio=self.i_ref/distance_px_tl
            else:
                self.ratio=None
        if self.mode_ratio=="reference_tape":
            self.ratio=self.i_ref

        traits_distances={key:self.compute_distance(value,self.ratio) for 
                            key,value in self.traits_coordinates.items()}
        self.traits_distances=traits_distances


    def tail_traits(self,dim_to_calculate):
        """Calculates traits associated with the tail

        Parameters
        ----------
        dim_to_calculate : str
            -CPd caudal peduncle minimal depth
            -CFd caudal fin depth
        """
        try:
            
            Tai_box=self.yolo_boxes.get("Tai")
            bc=self.from_box_to_r_c(Tai_box)
            center_tail_row=int((bc[1]-bc[0])/2)+bc[0]
            center_tail_col=int((bc[3]-bc[2])/2)+bc[2]
            #ref_pixel=self.closed_contour[center_tail_row,center_tail_col]

            #extra only the tail, while
            #preserving dimensions.
            get_tail_img=np.zeros(self.closed_contour.shape,np.uint8)
            get_tail_img[bc[0]:bc[1],bc[2]:bc[3]]=\
                                self.closed_contour[bc[0]:bc[1],bc[2]:bc[3]]
            #plt.imshow(get_tail_img), plt.axis("on")
            #plt.show()
            #ref_pixel=stats.mode(self.closed_contour[bc[0]:bc[1],bc[2]:bc[3]], 
            #                    axis=None)
            #get most frequent value of binary pixel whithin the tail box
            #ref_pixel=np.ravel(ref_pixel)[0]
            #print("ref_pixel",ref_pixel)
            

            if(dim_to_calculate=="CPd"):
                try:
                    col_pos=bc[2]
                    rear_col_pos=bc[2]
                    row_values=get_tail_img[:,bc[2]]
                    index_row_up=[index for 
                                    index,row in enumerate(row_values) if row==1 and 
                                    index<=center_tail_row]
                    #print("index_row_up",index_row_up)
                    index_row_up.reverse()
                    find_index_upper_edge=self.simple_edge_detector(index_row_up)
                    

                    index_row_down=[index for 
                                    index,row in enumerate(row_values) if row==1 and 
                                    index>center_tail_row]
                    #print("index_row_down",index_row_down)
                    find_index_down_edge=self.simple_edge_detector(index_row_down)
                    #print("edges",find_index_upper_edge,find_index_down_edge)
                    row_pos      = index_row_up[find_index_upper_edge]
                    rear_row_pos = index_row_down[find_index_down_edge]
                    self.CPd_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos]
                except:
                    col_pos=bc[2]
                    rear_col_pos=bc[2]
                    row_pos=bc[0]
                    rear_row_pos=bc[1]
                    self.CPd_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos]
            if(dim_to_calculate=="CFd"):
                col_pos=bc[3]
                rear_col_pos=bc[3]
                row_pos=bc[0]
                rear_row_pos=bc[1]
                self.CFd_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos]
        except:
            pass

    def TL(self):
        """Calculates the total length of the fish.
        This particular measurement might be used as
        reference to transform from pixel space to
        physical world.
        """
        Mou_box=self.yolo_boxes.get("Mou")
        Tai_box=self.yolo_boxes.get("Tai")
        Eye_box=self.yolo_boxes.get("Eye")
        try:
            if self.boolean_Mou:
                bm=self.from_box_to_r_c(Mou_box)
                bt=self.from_box_to_r_c(Tai_box)
            else:
                bm=self.from_box_to_r_c(Eye_box)
                bt=self.from_box_to_r_c(Tai_box)

            row_pos=int((bm[1]-bm[0])/2)+bm[0]
            rear_row_pos=row_pos

            col_pos=bm[2]
            rear_col_pos=bt[3]
            self.TL_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos] 
        except:
            self.TL_pp=None

    def Mo(self):
        """Calculates Mo. Distance from the top of the mouth to
        the bottom of the head along the head depth axis
        """
        Mou_box=self.yolo_boxes.get("Mou")
        try:
            if self.boolean_Mou and self.Eh_pp:
                trans_mou_coor=self.from_box_to_r_c(Mou_box)
                center=int((trans_mou_coor[1]-trans_mou_coor[0])/2)
                row_pos=center+trans_mou_coor[0]
                col_pos=trans_mou_coor[2]
                rear_col_pos=trans_mou_coor[2]

                if self.Eh_pp[2]>trans_mou_coor[1]:
                    rear_row_pos=self.Eh_pp[2]
                else:
                    rear_row_pos=trans_mou_coor[1]
                self.Mo_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos]
            else:
                self.Mo_pp=None
        except:
            self.Mo_pp=None
    def Bd(self):
        """Calculates Bd. body depth.
        """
        #H,W=self.closed_contour.shape()

        max_sum_col=self.closed_contour.sum(axis=0).argmax()
        print('maximum col:',max_sum_col)
        list_row=[index for 
            index,row in enumerate(
                        self.closed_contour[:,max_sum_col])
                         if row==1]
        row_pos=list_row[0]
        col_pos=max_sum_col
        rear_row_pos=list_row[-1]
        rear_col_pos=max_sum_col
        self.Bd_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos]

    def Bl(self):
        """Calculates Bl.body standard length.
        """
        try:
            shift=20
            row_pos=self.Mo_pp[2]+shift
            rear_row_pos=self.Mo_pp[2]+shift
            col_pos=self.Mo_pp[1]
            rear_col_pos=self.CPd_pp[3]
            self.Bl_pp=[row_pos,col_pos,rear_row_pos,rear_col_pos]
        except:
            self.Bl_pp=None

    def fin_traits(self,dim_to_calculate):
        """Calculates traits associated with the fin

        Parameters
        ----------
        dim_to_calculate : str
            can be:
            -PFi distance between the insertion of the pectoral 
                 fin to the bottom of the body
            -PFb body depth at the level 
                 of the pectoral fin insertion
            -PFl pectoral fin length.
        """
        try:
            Fin_box=self.yolo_boxes.get("Fin")
            trans_fin_coor=self.from_box_to_r_c(Fin_box)

            #center_fin_row=int((trans_fin_coor[1]-trans_fin_coor[0])/2)+trans_fin_coor[0]
            #center_fin_col=int((trans_fin_coor[3]-trans_fin_coor[2])/2)+trans_fin_coor[2]

            col_fin=trans_fin_coor[2]
            #x,y=self.closed_contour.shape[0],self.closed_contour.shape[1]
            #ref_pixel=self.closed_contour[x//2,y//2]
            #ref_pixel=self.closed_contour[center_fin_row,center_fin_col]
            row_list_1=[index for index,each_row in 
                    enumerate(self.closed_contour[:,col_fin])
                                                if each_row==1 and
                                                index<=trans_fin_coor[0]]

            # In case the top row of yolo box
            # falls outside the body, use that
            # coordinate as edged.
            if len(row_list_1)!=0:
                row_list_1.reverse()
                A=row_list_1[self.simple_edge_detector(row_list_1)]
            else:
                A=trans_fin_coor[0]

            row_list_A=[index for index,each_row in 
                    enumerate(self.closed_contour[:,col_fin])
                                                if each_row==1 and
                                                index>=A]
            B=row_list_A[self.simple_edge_detector(row_list_A)]

            #print("A,B",A,B)

            #row_edge=self.simple_edge_detector(row_list)
            if dim_to_calculate=="PFi":
                delta_col=(trans_fin_coor[1]-trans_fin_coor[0])/2
                row_pos=int(trans_fin_coor[0]+delta_col)
                #row_pos=row_list[0]
                
                #rear_row_pos=int(trans_fin_coor[1]+delta_col)
                rear_row_pos=B
                col_pos=col_fin
                rear_col_pos=col_fin
                self.PFi_pp=[row_pos,
                            col_pos,
                            rear_row_pos,
                            rear_col_pos]
            if dim_to_calculate=="PFb":
                row_pos=A
                rear_row_pos=B
                col_pos=col_fin+50
                rear_col_pos=col_fin+50
                self.PFb_pp=[row_pos,
                            col_pos,
                            rear_row_pos,
                            rear_col_pos]
            if dim_to_calculate=="PFl":
                row_pos=trans_fin_coor[0]
                col_pos=trans_fin_coor[2]
                rear_row_pos=row_pos
                rear_col_pos=trans_fin_coor[3]
                self.PFl_pp=[row_pos,
                            col_pos,
                            rear_row_pos,
                            rear_col_pos]
        except Exception as e: 
            print(e)
            pass

    @staticmethod
    def simple_edge_detector(input_list):
        """calculates the index where a discontinuity
        occurs giving and list of positional integers
        pixesl.

        Parameters
        ----------
        input_list : list

        Returns
        -------
        row_edge : int
            where discontinuity occurs
        """
        row_edge=len(input_list)-1
        for i in range(len(input_list)-1):
            delta=abs(input_list[i]-input_list[i+1])
            if (delta != 1 ):
                row_edge=i
                break
        return row_edge



    def eye_traits(self,dim_to_calculate):
        """Calculates traits associated with the eye

        Parameters
        ----------
        dim_to_calculate : str
            can be:
            -Hd head depth along the vertical axis of the eye)
            -Eh distance between the center of the eye to the bottom of 
                the head
            -Ed eye diameter
        """
        try:
            Eye_box=self.yolo_boxes.get("Eye")
            be=self.from_box_to_r_c(Eye_box)

            center_eye_row=int((be[1]-be[0])/2)+be[0]
            center_eye_col=int((be[3]-be[2])/2)+be[2]

            col_pos=rear_col_pos=center_eye_col
            # use reference pixel in case the binary transformation
            # invert the black and white colors.
            
            #ref_pixel=stats.mode(self.closed_contour,axis=None)[0][0]
            #x,y=self.closed_contour.shape[0],self.closed_contour.shape[1]
            #ref_pixel=self.closed_contour[x//2,y//2]
            ref_pixel=self.closed_contour[center_eye_row,center_eye_col]
            #print("ref_pixel",ref_pixel)

            if(dim_to_calculate=="Hd"):
                row_list=[index for index,each_row in 
                        enumerate(self.closed_contour[:,center_eye_col]) if 
                                                each_row!=abs(ref_pixel-1) and
                                                index<=center_eye_row]
                #print("row_list",row_list)
                row_list.reverse()
                #print("row_list.reverse",row_list)
                row_edge=self.simple_edge_detector(row_list)
                #print("row_edge",row_edge)
                
                row_pos=row_list[row_edge]
                #print("row_pos",row_pos)
                rear_row_pos=center_eye_row
                self.Hd_pp=[row_pos,
                            col_pos,
                            rear_row_pos,
                            rear_col_pos]

            if(dim_to_calculate=="Eh"):
                #select filled pixes from the center of the eye to the
                #end of the contour.
                row_list=[index for index,each_row in 
                        enumerate(self.closed_contour[:,center_eye_col]) if 
                                                int(each_row)==ref_pixel and
                                                index>=center_eye_row]
                #detect the edge by a sudden jump in the indexes.
                row_edge=self.simple_edge_detector(row_list)

                row_list=row_list[:row_edge]
                #print("new row list",row_list)


                row_pos=center_eye_row
                rear_row_pos=row_list[-1]
                self.Eh_pp=[row_pos,
                            col_pos,
                            rear_row_pos,
                            rear_col_pos]

            if(dim_to_calculate=="Ed"):
                col_pos=center_eye_col-30
                rear_col_pos=center_eye_col-30
                row_pos=be[0]
                rear_row_pos=be[1]
                self.Ed_pp=[row_pos,
                            col_pos,
                            rear_row_pos,
                            rear_col_pos]
        except Exception as e: 
            print(e)
            pass

    @staticmethod
    def from_box_to_r_c(box):
        """
        Auxiliar static function to get a more convenient representation
        of the bounding boxes detected by yolo.
        """
        row0=box[0][1]
        row1=box[1][1]
        col0=box[0][0]
        col1=box[1][0]
        return [row0,row1,col0,col1]

    @staticmethod
    def compute_distance(coordinates,ratio):
        """Auxiliar function to transform pair coordinates
        into distance in physical world.
        """
        if coordinates and ratio:
            last_coor=np.array((coordinates[0],coordinates[1]))
            start_coor=np.array((coordinates[2],coordinates[3]))
            distance_px=np.linalg.norm(last_coor-start_coor)
            distance_length=distance_px*ratio
            return distance_length
        else:
            return None

