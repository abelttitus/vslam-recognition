#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 11:29:56 2020

@author: abel
"""

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
from PIL import Image

fx = 481.20  # focal length x
fy = -480.00  # focal length y
cx = 319.50  # optical center x
cy = 239.50 # optical center y
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
    depth =cv2.imread(depth_file)
    
    # if rgb.size != depth.size:
    #     raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")


    points=np.zeros((480,640,3),dtype=np.float64)
    for u in range(rgb.shape[1]):
        for v in range(rgb.shape[0]):
            
            Z = depth[v,u,0] / scalingFactor
            if Z==0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points[v,u,0]=X
            points[v,u,1]=Y
            points[v,u,2]=Z
    pcd_color=np.concatenate([points,rgb])
    pcd_color=pcd_color.reshape((640*480,6))
    pcds=pcd_color[pcd_color[:,0]!=0.0]
    print("Pcd shape after removing:",pcds.shape)
    pcd_o = o3d.geometry.PointCloud()
    pcd_o.points=o3d.utility.Vector3dVector(pcds[:,:3])
    pcd_o.colors=o3d.utility.Vector3dVector(pcds[:,3:])
    o3d.io.write_point_cloud("/home/abel/vslam-recognition/gpu_test.ply", pcd_o)


    


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

    
    generate_pointcloud(img_path,depth_path,'/home/abel/vslam-recognition/pcd_i_new.ply')
