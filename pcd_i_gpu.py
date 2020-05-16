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
    start=time.time();
    rgb = cv2.imread(rgb_file)
    rgb=rgb.astype(np.uint8)

    depth = cv2.imread(depth_file)
    depth = depth[:,:,0]
    depth=depth.astype(np.float32)
    
    #coords
    x=np.zeros((depth.shape[0],depth.shape[1],1))
    x=x.astype(np.float32)
    y=np.zeros((depth.shape[0],depth.shape[1],1))
    y=y.astype(np.float32)
    z=np.zeros((depth.shape[0],depth.shape[1],1))
    z=z.astype(np.float32)
    
    
    if rgb.shape != depth.shape:
        raise Exception("Color and depth image do not have the same resolution.")
    # if rgb.mode != "RGB":
    #     raise Exception("Color image is not in RGB format")
    # if depth.mode != "I":
    #     raise Exception("Depth image is not in intensity format")

    depth_gpu=gpuarray.to_gpu(depth)
    x_gpu=cuda.mem_alloc(x.nbytes)
    y_gpu=cuda.mem_alloc(y.nbytes)
    z_gpu=cuda.mem_alloc(z.nbytes)    
    
    cuda.memcpy_htod(x_gpu,x)
    cuda.memcpy_htod(y_gpu,y)
    cuda.memcpy_htod(z_gpu,z)
    
    
    
    mod=SourceModule("""
                     __global__ void vmap_kernel(float* depth,float *x,float* y,float* z){
                       int u = threadIdx.x + blockIdx.x * blockDim.x;
                       int v = threadIdx.y + blockIdx.y * blockDim.y; 
                       
                       int cols=640;
                       int rows=480;
                       float fx = 481.20f;
                       float fy = -480.00f ;
                        float cx = 319.50f ;
                        float cy = 239.50f;
                        float sf=5000.0f;
                       
                       if(u<cols && v<rows){
                               
                               float d= depth[u+cols*v]/sf;
                               if(d!=0.0f){
                                       x[u+cols*v] =(u - cx) * d/ fx;
                                       y[u+cols*v]= (v - cy) * d / fy;
                                       z[u+cols*v]= d;
                               }
                       }
                       }""")
                
    function=mod.get_function("vmap_kernel");
    function(depth_gpu,x_gpu,y_gpu,z_gpu,block=(32,8,1),grid=(20,60,1))
    cuda.memcpy_dtoh(x,x_gpu)
    cuda.memcpy_dtoh(y,y_gpu)
    cuda.memcpy_dtoh(z,z_gpu)
    
    coords=np.concatenate([x,y,z],axis=2)    
    end=time.time()
    print("Time taken to generate vmap:",end-start)
    
    points=[] 
    for u in range(rgb.shape[1]):
        for v in range(rgb.shape[0]):
            if coords[v,u,2]==0.0:
                continue
            points.append("%f %f %f %d %d %d 0\n"%(coords[v,u,0],coords[v,u,1],coords[v,u,2],rgb[v,u,0],rgb[v,u,1],rgb[v,u,2]))
    
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

    
    generate_pointcloud(img_path,depth_path,'/home/abel/vslam-recognition/pcd_i_newest.ply')
   
    