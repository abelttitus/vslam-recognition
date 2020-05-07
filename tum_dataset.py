#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 10:44:43 2020

@author: abel
"""


import numpy as np
import cv2
import sys

if __name__=='__main__':
    
    if(sys.argv[1]=='rgb'):
        rgb_file_path='/home/abel/rgbd_dataset_freiburg1_xyz/rgb.txt'
        rgb_file=open(rgb_file_path,'rt')
        for line in rgb_file:
            if line[0]=='#':
                continue
            content = line.split()
            time=content[0]
            img_file=content[1]
            break
        img_dir='/home/abel/rgbd_dataset_freiburg1_xyz/'
        rgb_img=cv2.imread(img_file)
        print('Timestamp:',time)
        print("Max Value:",np.max(rgb_img))
        print("Min Value:",np.min(rgb_img))
        print("Shape:",rgb_img.shape)
      
        cv2.imshow("Color",rgb_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        rgb_file.close()

    if(sys.argv[1]=='depth'):
        depth_file_path='/home/abel/rgbd_dataset_freiburg1_xyz/depth.txt'
        depth_file=open(depth_file_path,'rt')
        for line in depth_file:
            if line[0]=='#':
                continue
            content = line.split()
            time=content[0]
            depth_file=content[1]
            break
        depth_dir='/home/abel/rgbd_dataset_freiburg1_xyz/'
        depth_file=depth_dir+depth_file
        depth_img=cv2.imread(depth_file)
        print('Timestamp:',time)
        print("Max Value:",np.max(depth_img))
        print("Min Value:",np.min(depth_img))
        print("Shape:",depth_img.shape)
      
        cv2.imshow("Depth",depth_img)
        cv2.waitKey()
        cv2.destroyAllWindows()
        depth_file.close()
