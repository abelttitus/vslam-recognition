#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 15 12:40:34 2020

@author: abel
"""


import numpy as np
import cv2
import open3d as o3d
from icp import *
from PIL import Image
import time
from numba import njit

fx = 481.20  # focal length x
fy = -480.00  # focal length y
cx = 319.50  # optical center x
cy = 239.50 # optical center y
scalingFactor = 5000.0

@njit(parallel=True)
def gen_vmap(rgb,depth):
    points = []    
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v,u,:]
            Z = depth[v,u,0]/ scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            #points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    
def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = cv2.imread(rgb_file)
    depth = cv2.imread(depth_file)
    
    if rgb.shape != depth.shape:
        raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")


    points=gen_vmap(rgb,depth)
#     file = open(ply_file,"w")
#     file.write('''ply
# format ascii 1.0
# element vertex %d
# property float x
# property float y
# property float z
# property uchar red
# property uchar green
# property uchar blue
# property uchar alpha
# end_header
# %s
# '''%(len(points),"".join(points)))
#     file.close()
    


if __name__=='__main__':
    verbose=True;
    with open('/home/abel/dataset/associations.txt') as file: 
        for line in file:
            contents=line.split()
            if verbose:
                print("Image Number:",contents[0])
                print("File (Depth):",contents[1])
                print("File (RGB)",contents[3])
            break
    base_dir='/home/abel/dataset/'
    img_path=base_dir+contents[3]
    depth_path=base_dir+contents[1]

    start=time.time();
    generate_pointcloud(img_path,depth_path,'/home/abel/vslam-recognition/pcd_i_new.ply')
    end=time.time()
    
    print("Time taken to generate vmap:",end-start)