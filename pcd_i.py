#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 21:04:49 2020

@author: abel
"""


import numpy as np
import cv2
import open3d as o3d

if __name__=='__main__':
    verbose=True;
    with open('/home/abel/dataset/associations.txt') as file: 
        for line in file:
            contents=line.split()
            if verbose:
                print("Image Number:",contents[0])
                print("File (Depth):",contents[1])
                print("File (RGB)",contents[2])
            break
    base_dir='/home/abel/dataset/'
    img_path=base_dir+contents[2]
    depth_path=base_dir+contents[1]
    # img=cv2.imread(img_path)
    depth=cv2.imread(depth_path)
    
    print("Depth Shape",depth.shape)
    if depth.size==0:
        print("Image reading Mistake")
    fx = 481.20  # focal length x
    fy = -480.00  # focal length y
    cx = 319.50  # optical center x
    cy = 239.50 # optical center y
    
    factor = 5000.0 # for the 16-bit PNG files
    
    coords=np.zeros((480,640,3))
    coords.astype(np.float32)
    for y in range(480):
        for x in range(640):
            coords[y,x,2]=float(depth[y,x,0])/factor
            Z=coords[y,x,2]
            coords[y,x,1]=(x-cx)*Z/fx
            coords[y,x,0]=(y-cy)*Z/fy
            
    vertices=np.reshape(coords,(307200,3))
    pcd=o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(vertices)
    o3d.io.write_point_cloud('/home/abel/vslam-recognition/pcd_i.ply', pcd)
    o3d.visualization.draw_geometries([pcd])