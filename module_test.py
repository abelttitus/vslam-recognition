import open3d as o3d 
import cv2
import numpy as np

if __name__ =='__main__':
   rgb_img=cv2.imread('/home/abel/dataset/rgb/1.png') 
   print("Max Value:",np.max(rgb_img))
   print("Min Value:",np.min(rgb_img))
   print("Shape:",rgb_img.shape)
   
   depth_img=cv2.imread('/home/abel/dataset/depth/1.png') 
   print("Max Value:",np.max(depth_img))
   print("Min Value:",np.min(depth_img))
   print("Shape:",depth_img.shape) 