#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:04:49 2020

@author: abel
"""


import numpy as np
import cv2
import open3d as o3d
from icp import *

if __name__=='__main__':
    verbose=True;
    base_dir='/home/abel/dataset/'
    fx = 481.20  # focal length x
    fy = -480.00  # focal length y
    cx = 319.50  # optical center x
    cy = 239.50 # optical center y
    factor = 5000.0 # for the 16-bit PNG files
    coords=np.zeros((480,640,3))
    coords.astype(np.float32)
    pcds=[]
    

    with open('/home/abel/dataset/associations.txt') as file: 
        for line in file:
            contents=line.split()
            if verbose:
                print("Image Number:",contents[0])
                print("File (Depth):",contents[1])
                print("File (RGB)",contents[2])
            
            img_no=int(contents[0])
            img_path=base_dir+contents[2]
            depth_path=base_dir+contents[1]
            # img=cv2.imread(img_path)
            depth=cv2.imread(depth_path)
            
            print("Depth Shape",depth.shape)
            if depth.size==0:
                print("Image reading Mistake")

    
    
            for y in range(480):
                for x in range(640):
                    coords[y,x,2]=float(depth[y,x,0])/factor
                    Z=coords[y,x,2]
                    coords[y,x,1]=(x-cx)*Z/fx
                    coords[y,x,0]=(y-cy)*Z/fy
                    
            vertices=np.reshape(coords,(307200,3))
            
            pcds.append(vertices)
        
            if verbose:
                print("Length of pcds after appending ",img_no, "is",len(pcds))
                print("Shape of ",img_no,pcds[img_no].shape)
                print("Unique in pcds",img_no,np.unique(pcds[img_no]))
        
            if img_no==2:
                break
    