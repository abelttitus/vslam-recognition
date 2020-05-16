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
    rgb=rgb.astype(np.float32)
    rgb = cv2.normalize(rgb, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    print(np.max(rgb))
    print(np.min(rgb))
    depth = cv2.imread(depth_file)
    depth=depth.astype(np.float32)
    
    #coords
    x=np.zeros((depth.shape[0],depth.shape[1]))
    x=x.astype(np.float32)
    y=np.zeros((depth.shape[0],depth.shape[1]))
    y=y.astype(np.float32)
    z=np.zeros((depth.shape[0],depth.shape[1]))
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
                       float scaling_factor=5000.0;
                       float cx=319.50;
                       float cy=239.50;
                       float fx_inv=1/481.20;
                       float fy_inv=-1/480.00;
                       
                       if(u<cols && v<rows){
                               
                               float d= depth[u+cols*v]/scaling_factor;
                               if(d!=0){
                                       x[u+cols*v] =1.0;
                                       y[u+cols*v]= 2.0;
                                       z[u+cols*v]= 3.0;
                               }
                       }
                       }""")
                
    function=mod.get_function("vmap_kernel");
    function(depth_gpu,x_gpu,y_gpu,z_gpu,block=(32,8,1),grid=(20,60,1))
    cuda.memcpy_dtoh(x,x_gpu)
    cuda.memcpy_dtoh(y,y_gpu)
    cuda.memcpy_dtoh(z,z_gpu)
    
    points=np.concatenate([x,y,z],axis=2)
    print("Random values in points along z axis",points[40,50,:],points[32,90,:])
    pcd_color=np.concatenate([points,rgb])
    pcd_color=pcd_color.reshape((640*480,6))
    pcds=pcd_color[pcd_color[:,0]!=0.0]
    pcd_o = o3d.geometry.PointCloud()
    pcd_o.points=o3d.utility.Vector3dVector(pcds[:,:3])
    pcd_o.colors=o3d.utility.Vector3dVector(pcds[:,3:])
    o3d.io.write_point_cloud("/home/abel/vslam-recognition/gpu.ply", pcd_o)


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