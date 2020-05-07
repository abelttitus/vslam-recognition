#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 12:15:54 2020

@author: abel
"""


import numpy as np
import cv2
import open3d as o3d

if __name__=='__main__':
    verbose=True;
    with open('/home/abel/rgbd_dataset_freiburg1_xyz/depth.txt') as d_file, open('/home/abel/rgbd_dataset_freiburg1_xyz/rgb.txt') as rgb_file: 
        for x, y in zip(rgb_file, d_file):
            x = x.strip()
            y = y.strip()
            if(x[0]=='#'):
                continue
            x_line=x.split()
            y_line=y.split()
            if verbose:
                print("Time (RGB):",x_line[0])
                print("File (RGB):",x_line[1])
                print("Time (Depth)",y_line[0])
                print("File (Depth)",y_line[1])
            break
    base_dir='/home/abel/rgbd_dataset_freiburg1_xyz/'
    img_path=base_dir+x_line[1]
    depth_path=base_dir+y_line[1]
    img=cv2.imread(img_path)
    depth=cv2.imread(depth_path)
    

    fx = 525.0  # focal length x
    fy = 525.0  # focal length y
    cx = 319.5  # optical center x
    cy = 239.5  # optical center y
    
    factor = 5000.0 # for the 16-bit PNG files
    
    coords=np.zeros((480,640,3))
    coords.astype(np.float32)
    for y in range(480):
        for x in range(640):
            coords[y,x,2]=float(depth[y,x,0])/factor
            Z=coords[y,x,2]
            coords[y,x,1]=(x-cx)*Z/fx
            coords[y,x,0]=(y-cy)*Z/fy
    print(np.unique(coords))      
    print(np.unique(depth))
