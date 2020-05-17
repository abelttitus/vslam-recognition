#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:17:38 2020

@author: abel
"""
import cv2
import numpy as np
import sys

if __name__ =='__main__':
    
   img_no=sys.argv[1]
   depth_img=cv2.imread('/home/abel/dataset/depth/'+str(img_no)+'.png') 
   print("Max Value:",np.max(depth_img))
   print("Min Value:",np.min(depth_img))
   print("Shape:",depth_img.shape) 
   
   cv2.imshow("Depth",depth_img)
   cv2.waitKey()
   cv2.destroyAllWindows()
   
   rgb=cv2.imread('/home/abel/dataset/rgb/'+str(img_no)+'.png') 
   print("Max Value:",np.max(rgb))
   print("Min Value:",np.min(rgb))
   print("Shape:",rgb.shape) 
   
   cv2.imshow("Color",rgb)
   cv2.waitKey()
   cv2.destroyAllWindows()