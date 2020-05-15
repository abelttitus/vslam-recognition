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


import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray


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
    depth = cv2.imread(depth_file)
    depth=depth.astype(np.float32)
    points=np.zeros((depth.shape[0],depth.shape[1],3))
    points=points.astype(np.float32)
    
    if rgb.shape != depth.shape:
        raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")

    depth_gpu=gpuarray.to_gpu(depth)
    points_gpu=cuda.mem_alloc(points.nbytes)
    cuda.memcpy_htod(points_gpu,points)
    
    
    mod=SourceModule("""
                     __global__ void vmap_kernel(float* depth,float* points){
                       int u = threadIdx.x + blockIdx.x * blockDim.x;
                       int v = threadIdx.y + blockIdx.y * blockDim.y; 
                       
                       int cols=640;
                       int rows=480;
                       float scaling_factor=5000.0;
                       float depthCutoff=20.0f;
                       float cx=319.50;
                       float cy=239.50;
                       float fx_inv=1/481.20;
                       float fy_inv=-1/480.00;
                       
                       if(u<cols && v<rows){
                               float z= depth[u+cols*v]/scaling_factor;
                               if(z!=0 && z<depthCutoff){
                                       float vx = z * (u - cx) * fx_inv;
                                        float vy = z * (v - cy) * fy_inv;
                                        float vz = z;
                                       
                                        points[u+cols*v]=vx;
                                        points[u+cols*v+1]=vy;
                                        points[u+cols*v+2]=vz;
                                        
                                        
                               }
                       }
                       }""")
                
    function=mod.get_function("vmap_kernel");
    function(depth_gpu,points_gpu,block=(32,8,1),grid=(20,60,1))
    cuda.memcpy_dtoh(points,points_gpu)
    print(points.shape)


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