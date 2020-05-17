#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:10:19 2020

@author: abel
"""

import numpy as np
import cv2
import open3d as o3d
from icp import *
from PIL import Image

fx = 481.20  # focal length x
fy = -480.00  # focal length y
cx = 319.50  # optical center x
cy = 239.50 # optical center y
scalingFactor = 5000.0


def generate_pointcloud(rgb_file,depth_file,ply_file,img_no):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.
    
    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file
    
    """
    rgb = cv2.imread(rgb_file)
    rgb=rgb.astype(np.uint8)
    depth =cv2.imread(depth_file)
    
    # if rgb.size != depth.size:
    #     raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")

    cols=rgb.shape[1]
    rows=rgb.shape[0];
    
    
    v_map=np.zeros((480,640,3),dtype=np.float32)
    for u in range(cols):
        for v in range(rows):
            
            Z = depth[v,u,0] / scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            v_map[v,u,0]=X
            v_map[v,u,1]=Y
            v_map[v,u,2]=Z
       
    n_map=np.zeros((480,640,3),dtype=np.float32)
    for u in range(cols):
        for v in range(rows):
            if(u==cols-1 or v==rows-1):
                continue
            
            v00=v_map[v,u,:]
            v01=v_map[v,u+1,:]
            v10=v_map[v+1,u,:]
            
            if(v00[2]==0.0 or v01[2]==0.0 or v10[2]==0.0):
                continue
            
            p=v01-v00
            q=v10-v00
            norm=np.cross(p,q)
            norm=norm/np.linalg.norm(norm)
            n_map[v,u,:]=norm
            
    np.save('/home/abel/vslam-recognition/np/v_map'+str(img_no),v_map)
    np.save('/home/abel/vslam-recognition/np/n_map'+str(img_no),n_map)
    
     
    
#     points=[] 
#     for u in range(cols):
#         for v in range(rows):
#             if coords[v,u,2]==0.0 or n_map[v,u,0]==0:
#                 continue
#             points.append("%f %f %f %d %d %d 0\n"%(coords[v,u,0],coords[v,u,1],coords[v,u,2],n_map[v,u,0],n_map[v,u,1],n_map[v,u,2]))
    
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
            img_no=int(contents[0])
            base_dir='/home/abel/dataset/'
            img_path=base_dir+contents[3]
            depth_path=base_dir+contents[1]
        
            if img_no<3 and img_no!=0:
                generate_pointcloud(img_path,depth_path,'/home/abel/vslam-recognition/pcd_my_norm.ply',img_no)
                print("Gen vmap and nmap of",img_no)