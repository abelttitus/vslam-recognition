#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 22:48:29 2020

@author: abel
"""


import numpy as np
from native_icp import *

fx = 481.20  # focal length x
fy = -480.00  # focal length y
cx = 319.50  # optical center x
cy = 239.50 # optical center y

def pcd_transform(trans,points):
    #points N*3
    N=points.shape[0]
    ones=np.ones((N,1),dtype=np.float32)
    homo_points=np.concatenate([points,ones],axis=1)
    p_t=np.matmul(trans,np.transpose(homo_points))
    return np.transpose(p_t)[:,:3]


vmap_src=np.load('/home/abel/vslam-recognition/np/v_map2.npy')
vmap_dst=np.load('/home/abel/vslam-recognition/np/v_map1.npy')
nmap_src=np.load('/home/abel/vslam-recognition/np/n_map2.npy')
nmap_dst=np.load('/home/abel/vslam-recognition/np/n_map1.npy')


pose_1=np.array([[-0.999762,0.000000,-0.021799,1.370500],[
0.000000,1.000000,0.000000,1.517390],[
0.021799,0.000000,-0.999762,1.449630],[
    0.0,0.0,0.0,1.0]])
 
pose_2=np.array([[-0.999738,-0.000418,-0.022848,1.370020],[
-0.000464,0.999998,0.002027,1.526344],[
0.022848,0.002037,-0.999737,1.448990],
    [0.0,0.0,0.0,1.0]])
    
cols=640
rows=480
angle_thresh=np.sin(20. * 3.14159254 / 180.)
dist_thresh=0.10
init_pose=np.identity(4)

count=0


u=50
v=40

for u in range(cols):
    for v in range(rows):

        v_src=np.expand_dims(vmap_src[v,u],axis=0)
        v_src_in_dst=pcd_transform(init_pose,v_src)
        x=(v_src_in_dst[0,0]*fx+v_src_in_dst[0,2]*cx)/v_src_in_dst[0,2]
        y=(v_src_in_dst[0,1]*fy+v_src_in_dst[0,2]*cy)/v_src_in_dst[0,2]
        x=int(np.round(x))
        y=int(np.round(y))
        
        if x>=0 and y>=0 and x<cols and y<rows and v_src[0,2]>0 and v_src_in_dst[0,2]>0:
            v_dst=np.expand_dims(vmap_dst[y,x],axis=0)
            n_src=np.expand_dims(nmap_src[v,u],axis=0)
            n_dst=np.expand_dims(nmap_dst[y,x],axis=0)
            n_src_in_dst=pcd_transform(init_pose,n_src)
            
            angle=np.linalg.norm(np.cross(n_src_in_dst,n_dst))
            dist=np.linalg.norm(v_src_in_dst-v_dst)
            
            if angle<angle_thresh and dist<dist_thresh and np.linalg.norm(n_src)!=0 and np.linalg.norm(n_dst)!=0 and v_src.all()!=v_dst.all():
                count+=1
                print(u,v)
                    
print("No of Correspondances found:",count)
                
