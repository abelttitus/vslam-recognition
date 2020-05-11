#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 11 13:18:04 2020

@author: abel
"""


import numpy as np
import cv2
import open3d as o3d
from PIL import Image

focalLength = 525.0
centerX = 319.5
centerY = 239.5
scalingFactor = 5000.0


def generate_pointcloud(rgb_file,depth_file):
    """
    Generate a colored point cloud in Numpy array (N*3) from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image

    
    """
    rgb = Image.open(rgb_file)
    depth = Image.open(depth_file)
    
    if rgb.size != depth.size:
        raise Exception("Color and depth image do not have the same resolution.")
    if rgb.mode != "RGB":
        raise Exception("Color image is not in RGB format")
    if depth.mode != "I":
        raise Exception("Depth image is not in intensity format")


    points = np.zeros((rgb.size[1],rgb.size[0],3))
    valid=np.zeros((rgb.size[1]*rgb.size[0]),dtype=bool)
    
    for v in range(rgb.size[1]):
        for u in range(rgb.size[0]):
            color = rgb.getpixel((u,v))
            Z = depth.getpixel((u,v)) / scalingFactor
            if Z==0: continue
            X = (u - centerX) * Z / focalLength
            Y = (v - centerY) * Z / focalLength
            points[v,u,0]=X
            points[v,u,1]=Y
            points[v,u,2]=Z
            valid[v*rgb.size[1]+u]=True

    vertices=np.reshape(points(rgb.size[1]*rgb.size[0],3))
    
    return vertices[valid]


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

    coords=generate_pointcloud(img_path,depth_path)
    print("PCD Shape:",coords.shape)