import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
import math
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from tqdm import tqdm

def get_rot_from_yaw(yaw):
    cy=math.cos(yaw)
    sy=math.sin(yaw)
    rot=np.array([[cy,0,sy],
                  [0,1,0],
                  [-sy,0,cy]])
    return rot
def get_rot_from_roll(roll):
    cr = math.cos(roll)
    sr = math.sin(roll)
    rot=np.array([[cr,-sr,0],
                  [sr,cr,0],
                  [0,0,1]])
    return rot
def get_rot_from_pitch(pitch):
    cp=math.cos(pitch)
    sp=math.sin(pitch)
    rot=np.array([
        [1,0,0],
        [0,cp,-sp],
        [0,sp,cp]
    ])
    return rot

def get_rot_matrix(yaw,roll,pitch):
    yaw_rot=get_rot_from_yaw(yaw)
    roll_rot=get_rot_from_roll(roll)
    pitch_rot=get_rot_from_pitch(pitch)
    rot_mat=np.dot(np.dot(yaw_rot,roll_rot),pitch_rot)
    return rot_mat

@DATASETS.register_module()
class ScanObjectNN(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.use_aug = config.use_aug
        self.pre_aug = config.pre_aug
        self.root = config.ROOT
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        if self.use_aug and self.pre_aug:
            for i, _points in tqdm(enumerate(self.points)):
                yaw=np.random.random()*np.pi/2-np.pi/4
                roll=np.random.random()*np.pi/2-np.pi/4
                pitch=np.random.random()*np.pi/2-np.pi/4
                rot_mat=get_rot_matrix(yaw,roll,pitch)
                self.points[i]=np.dot(_points,rot_mat.T)

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')

    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        
        if self.use_aug and not self.pre_aug:
            yaw=np.random.random()*np.pi/2-np.pi/4
            roll=np.random.random()*np.pi/2-np.pi/4
            pitch=np.random.random()*np.pi/2-np.pi/4
            rot_mat=get_rot_matrix(yaw,roll,pitch)
            current_points=np.dot(current_points,rot_mat.T)

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]



@DATASETS.register_module()
class ScanObjectNN_hardest(Dataset):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        self.use_aug = config.use_aug
        self.pre_aug = config.pre_aug
        
        if self.subset == 'train':
            h5 = h5py.File(os.path.join(self.root, 'training_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        elif self.subset == 'test':
            h5 = h5py.File(os.path.join(self.root, 'test_objectdataset_augmentedrot_scale75.h5'), 'r')
            self.points = np.array(h5['data']).astype(np.float32)
            self.labels = np.array(h5['label']).astype(int)
            h5.close()
        else:
            raise NotImplementedError()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape}')
        
        if self.use_aug and self.pre_aug:
            for i, _points in tqdm(enumerate(self.points)):
                yaw=np.random.random()*np.pi/2-np.pi/4
                roll=np.random.random()*np.pi/2-np.pi/4
                pitch=np.random.random()*np.pi/2-np.pi/4
                rot_mat=get_rot_matrix(yaw,roll,pitch)
                self.points[i]=np.dot(_points,rot_mat.T)
    
    def __getitem__(self, idx):
        pt_idxs = np.arange(0, self.points.shape[1])   # 2048
        if self.subset == 'train':
            np.random.shuffle(pt_idxs)
        
        current_points = self.points[idx, pt_idxs].copy()
        
        if self.use_aug and not self.pre_aug:
            yaw=np.random.random()*np.pi/2-np.pi/4
            roll=np.random.random()*np.pi/2-np.pi/4
            pitch=np.random.random()*np.pi/2-np.pi/4
            rot_mat=get_rot_matrix(yaw,roll,pitch)
            current_points=np.dot(current_points,rot_mat.T)

        current_points = torch.from_numpy(current_points).float()
        label = self.labels[idx]
        
        return 'ScanObjectNN', 'sample', (current_points, label)

    def __len__(self):
        return self.points.shape[0]