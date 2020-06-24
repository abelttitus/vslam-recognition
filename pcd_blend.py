#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 15:36:15 2020

@author: abel
"""


import numpy as np

import open3d as o3d
from PIL import Image
import sys

sys.path[3]=''
import cv2
scalingFactor = 5000.0


def generate_pointcloud(rgb_file,depth_file,ply_file):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = cv2.imread(rgb_file)
    depth = cv2.imread(depth_file,cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    
    K=np.loadtxt('/home/abel/blender/blend_K_bottle.txt')
    fx=K[0,0]
    fy=K[1,1]
    cx=K[0,2]
    cy=K[1,2]



    points = []    
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v,u]
            Z = depth[v,u] / scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n"%(X,Y,Z,color[0],color[1],color[2]))
    file = open(ply_file,"w")
    file.write('''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
property uchar alpha
end_header
%s
'''%(len(points),"".join(points)))
    file.close()
    


if __name__=='__main__':

    img_path='/home/abel/blender/output_bottle/rgb/Image_0000.png'
    depth_path='/home/abel/blender/output_bottle/depth/0.png'

    generate_pointcloud(img_path,depth_path,'/home/abel/blender/bottle_0.ply')