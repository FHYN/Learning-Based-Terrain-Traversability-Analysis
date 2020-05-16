import os
import cv2
import json
import yaml
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
import threading
import multiprocessing
from PIL import Image

from .fhy4_datautils_test import Semantic_KITTI_Utils
#from redis_utils import Mat_Redis_Utils

def pcd_jitter(pcd, sigma=0.01, clip=0.05):
    N, C = pcd.shape 
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1*clip, clip).astype(pcd.dtype)
    jittered_data += pcd
    return jittered_data

def pcd_normalize(pcd):
    pcd = pcd.copy()
    pcd[:,0] = pcd[:,0] / 70
    pcd[:,1] = pcd[:,1] / 70
    pcd[:,2] = pcd[:,2] / 3
    pcd[:,3] = (pcd[:,3] - 0.5)*2
    pcd = np.clip(pcd,-1,1)
    return pcd

def pcd_unnormalize(pcd):
    pcd = pcd.copy()
    pcd[:,0] = pcd[:,0] * 70
    pcd[:,1] = pcd[:,1] * 70
    pcd[:,2] = pcd[:,2] * 3
    pcd[:,3] = pcd[:,3] / 2 + 0.5
    return pcd

def pcd_tensor_unnorm(pcd):
    pcd_unnorm = pcd.clone()
    pcd_unnorm[:,0] = pcd[:,0] * 70
    pcd_unnorm[:,1] = pcd[:,1] * 70
    pcd_unnorm[:,2] = pcd[:,2] * 3
    pcd_unnorm[:,3] = pcd[:,3] / 2 + 0.5
    return pcd_unnorm

class SemKITTI_Loader(Dataset):
    def __init__(self, root, npoints, train = True, subset = 'inview'):
        self.root = root
        self.train = train
        self.npoints =npoints
        self.load_image = not train
        self.utils = Semantic_KITTI_Utils(root, subset)
        # self.np_redis = Mat_Redis_Utils()
        part_length = {'00': 1811,'01':1543,'02':1707,'03':1934}
        self.keys = []
        if self.train:
            for part in ['00','02','03']:
                length = part_length[part]
                for index in range(0,length,3):
                    self.keys.append('%s/%06d'%(part, index))
        else:
            for part in ['01']:
                length = part_length[part]
                for index in range(0,length,3):
                    self.keys.append('%s/%06d'%(part, index))        

    def __len__(self):
            return len(self.keys)

    def get_data(self, key):
        part, index = key.split('/')
        point_cloud, label = self.utils.get_pts_l(part, int(index))
        return point_cloud, label

    def __getitem__(self, index):
        point_cloud, label = self.get_data(self.keys[index])
        pcd = pcd_normalize(point_cloud)
        if self.train:
            pcd = pcd_jitter(pcd)
        # length = pcd.shape[0]
        # if length == self.npoints:
        #     pass
        # elif length > self.npoints:
        #     start_idx = np.random.randint(0, length - self.npoints)
        #     end_idx = start_idx + self.npoints
        #     pcd = pcd[start_idx:end_idx]
        #     label = label[start_idx:end_idx]
        # else:
        #     rows_short = self.npoints - length
        #     pcd = np.concatenate((pcd,pcd[0:rows_short]),axis=0)
        #     label = np.concatenate((label,label[0:rows_short]),axis=0)

        length = pcd.shape[0]
        choice = np.random.choice(length, self.npoints, replace=True)
        pcd = pcd[choice]
        label = label[choice]
        return pcd, label


if __name__ == '__main__':
    data_path = "/home/james/fhy/img_velo_label_327"
    s = SemKITTI_Loader(data_path, train = True, subset = 'inview')
    pcd, label = s[0]
    print(pcd.dtype,label.shape)

