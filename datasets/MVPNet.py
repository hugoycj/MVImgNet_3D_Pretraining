from copyreg import pickle
import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
import pickle
from utils.logger import *
import math
import glob

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

def get_rot_matrix(yaw,roll,pitch):
    yaw_rot=get_rot_from_yaw(yaw)
    roll_rot=get_rot_from_roll(roll)
    pitch_rot=get_rot_from_pitch(pitch)
    rot_mat=np.dot(np.dot(yaw_rot,roll_rot),pitch_rot)
    return rot_mat

def save_point2ply(points,save_path):
    ply_mesh=trimesh.Trimesh()
    ply_mesh.vertices=np.asarray(points)
    ply_mesh.export(save_path)


@DATASETS.register_module()
class MVPNet(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.subset = config.subset
        self.npoints = config.N_POINTS
        self.sample_points_num = config.npoints
        self.permutation = np.arange(self.npoints)
        self.use_color=config.USE_COLOR
        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'MVPNet')
        
        
        self.point_cloud_list=[]
        self.label_list=[]
        self.filename_list=[]
        data_list=glob.glob(os.path.join(self.data_root,self.subset, "*.pkl"))
        for datapath in data_list:
            print_log(f'[DATASET] {datapath} is opened', logger = 'MVPNet')

            with open(datapath,'rb') as f:
                content=pickle.load(f)
            self.point_cloud_list+=content["point_cloud"]
            self.label_list+=content["label"]
            self.filename_list+=content["filename"]
        print_log(f'[DATASET] {len(self.filename_list)} instances were loaded', logger = 'MVPNet')


        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        data  = self.point_cloud_list[idx]
        class_id = self.label_list[idx]
        filename = self.filename_list[idx]
        
        if not self.use_color:
            data=data[:,0:3]

        data = self.random_sample(data, self.sample_points_num)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return class_id, filename, data

    def __len__(self):
        return len(self.filename_list)


