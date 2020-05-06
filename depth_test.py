#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:17:38 2020

@author: abel
"""
import cv2
import numpy as np


if __name__ =='__main__':
   depth_img=cv2.imread('/home/abel/dataset/depth/1.png') 
   print("Max Value:",np.max(depth_img))
   print("Min Value:",np.min(depth_img))
   print("Shape:",depth_img.shape) 
   
   cv2.imshow("Depth",depth_img)