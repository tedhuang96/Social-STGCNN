# data_massage.py
import sys
pkg_path = '.'
sys.path.append(pkg_path)
import os
import pickle
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import torch

import os

import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx

from utils import * 
from metrics import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

import matplotlib.pyplot as plt


def arg_parse():
    parser = argparse.ArgumentParser()

    #Model specific parameters
    parser.add_argument('--input_size', type=int, default=2)
    parser.add_argument('--output_size', type=int, default=5)
    parser.add_argument('--n_stgcnn', type=int, default=1,help='Number of ST-GCNN layers')
    parser.add_argument('--n_txpcnn', type=int, default=5, help='Number of TXPCNN layers')
    parser.add_argument('--kernel_size', type=int, default=3)

    #Data specifc paremeters
    parser.add_argument('--obs_seq_len', type=int, default=8)
    parser.add_argument('--pred_seq_len', type=int, default=12)
    parser.add_argument('--dataset', default='eth',
                        help='eth,hotel,univ,zara1,zara2')    

    #Training specifc parameters
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--num_epochs', type=int, default=250,
                        help='number of epochs')  
    parser.add_argument('--clip_grad', type=float, default=None,
                        help='gadient clipping')        
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_sh_rate', type=int, default=150,
                        help='number of steps to drop the lr')  
    parser.add_argument('--use_lrschd', action="store_true", default=False,
                        help='Use lr rate scheduler')
    parser.add_argument('--tag', default='tag',
                        help='personal tag for the model ')
    return parser.parse_args()

  
args = arg_parse()
print('*'*30)
print("Training initiating....")
print(args)


# def graph_loss(V_pred,V_target):
#     return bivariate_loss(V_pred,V_target)

# #Data prep     
# obs_seq_len = args.obs_seq_len
# pred_seq_len = args.pred_seq_len
# data_set = './datasets/'+args.dataset+'/'

# dset_train = TrajectoryDataset(
#         data_set+'train/',
#         obs_len=obs_seq_len,
#         pred_len=pred_seq_len,
#         skip=1,norm_lap_matr=True)

# loader_train = DataLoader(
#         dset_train,
#         batch_size=1, #This is irrelative to the args batch size parameter
#         shuffle =True,
#         num_workers=0)


#Data prep     
obs_seq_len = args.obs_seq_len
pred_seq_len = args.pred_seq_len
data_set = './datasets/'+args.dataset+'/'

dset_test = TrajectoryDataset(
        data_set+'test/',
        obs_len=obs_seq_len,
        pred_len=pred_seq_len,
        skip=1,norm_lap_matr=True)

loader_test = DataLoader(
        dset_test,
        batch_size=1,#This is irrelative to the args batch size parameter
        shuffle =False,
        num_workers=1)


def get_batch_sample(loader_test):
    batch_count = 1
    for cnt,batch in enumerate(loader_test): 
        batch_count+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
            loss_mask,V_obs,A_obs,V_tr,A_tr = batch
        var_names = ['obs_traj', 'pred_traj_gt', 'obs_traj_rel', 'pred_traj_gt_rel', 'non_linear_ped',\
            'loss_mask','V_obs','A_obs','V_tr','A_tr']
        # print('-'*50)
        for var_name_i, batch_i in zip(var_names, batch):
            if var_name_i == 'obs_traj' and batch_i.shape[1]==4:
                return batch
        #     print(var_name_i)
        #     # print(type(batch_i))
        #     print(batch_i.shape)
        # print('-'*50)
    # break

batch = get_batch_sample(loader_test)
obs_traj, pred_traj_gt, obs_traj_rel, pred_traj_gt_rel, non_linear_ped,\
    loss_mask,V_obs,A_obs,V_tr,A_tr = batch

print(non_linear_ped)
print(loss_mask)
# obs_traj
# torch.Size([1, 3, 2, 8])
# pred_traj_gt
# torch.Size([1, 3, 2, 12])
# obs_traj_rel
# torch.Size([1, 3, 2, 8])
# pred_traj_gt_rel
# torch.Size([1, 3, 2, 12])

for i in range(4):
    try_obs_traj_rel = obs_traj[0,i,:, 1:]-obs_traj[0,i,:, :-1]
    # print(obs_traj[0,0])
    # print(obs_traj[0,0,:, 1:]-obs_traj[0,0,:, :-1])
    # print(obs_traj_rel[0,0])
    print(((try_obs_traj_rel-obs_traj_rel[0,i,:,1:])**2.).sum())


    try_gt_rel = torch.cat((obs_traj[0,i,:,-1:], pred_traj_gt[0,i]),dim=1) # (2,13)
    try_gt_rel = try_gt_rel[:,1:]-try_gt_rel[:,:-1]
# print(pred_traj_gt[0,0])
# print(pred_traj_gt[0,0,:, 1:]-pred_traj_gt[0,0,:, :-1])
# print(pred_traj_gt_rel[0,0])
    print(((try_gt_rel-pred_traj_gt_rel[0,i])**2.).sum())

# tensor([[6.1600, 5.7800, 5.4000, 4.9900, 4.5900, 4.1300, 3.7300, 3.3000],
#         [7.7600, 7.7300, 7.6900, 7.7200, 7.7100, 7.7800, 7.9200, 7.9700]],
#        device='cuda:0')
# tensor([[-0.3800, -0.3800, -0.4100, -0.4000, -0.4600, -0.4000, -0.4300],
#         [-0.0300, -0.0400,  0.0300, -0.0100,  0.0700,  0.1400,  0.0500]],
#        device='cuda:0')
# tensor([[ 0.0000, -0.3800, -0.3800, -0.4100, -0.4000, -0.4600, -0.4000, -0.4300],
#         [ 0.0000, -0.0300, -0.0400,  0.0300, -0.0100,  0.0700,  0.1400,  0.0500]],
#        device='cuda:0')


# obs_traj
# torch.Size([1, 3, 2, 8])
# pred_traj_gt
# torch.Size([1, 3, 2, 12])
# obs_traj_rel
# torch.Size([1, 3, 2, 8])
# pred_traj_gt_rel
# torch.Size([1, 3, 2, 12])

# try_gt_rel = torch.cat((obs_traj[0,0,:,-1:], pred_traj_gt[0,0]),dim=1) # (2,13)
# try_gt_rel = try_gt_rel[:,1:]-try_gt_rel[:,:-1]
# # print(pred_traj_gt[0,0])
# # print(pred_traj_gt[0,0,:, 1:]-pred_traj_gt[0,0,:, :-1])
# print(pred_traj_gt_rel[0,0])
# print(((try_gt_rel-pred_traj_gt_rel[0,0])**2.).sum())
fig, ax = plt.subplots()
for ped_i in range(obs_traj.shape[1]):
    x_obs_i = obs_traj[0, ped_i].to('cpu')
    x_gt_i = pred_traj_gt[0, ped_i].to('cpu')
    ax.plot(x_obs_i[0,:], x_obs_i[1,:], 'k.-')
    ax.plot(x_gt_i[0,:], x_gt_i[1,:], 'r.-')
plt.show()
# for sample_pred in x_pred:
#     plt.plot(sample_pred[:,0], sample_pred[:,1], 'c-')
# plt.plot(x_obs[:, 0], x_obs[:, 1], 'g-')
# plt.plot(x_gt[:, 0],x_gt[:, 1], 'r-')
# plt.xlim(-1, 16)
# plt.ylim(-1, 14)
# plt.show()

# var_names = ['obs_traj', 'pred_traj_gt', 'obs_traj_rel', 'pred_traj_gt_rel', 'non_linear_ped',\
#     'loss_mask','V_obs','A_obs','V_tr','A_tr']

# for var_name_i, batch_i in zip(var_names, batch):
#     print(var_name_i)
#     print(batch_i.shape)

# obs_traj
# torch.Size([1, 3, 2, 8])
# pred_traj_gt
# torch.Size([1, 3, 2, 12])
# obs_traj_rel
# torch.Size([1, 3, 2, 8])
# pred_traj_gt_rel
# torch.Size([1, 3, 2, 12])
# non_linear_ped
# torch.Size([1, 3])
# loss_mask
# torch.Size([1, 3, 20])
# V_obs
# torch.Size([1, 8, 3, 2])
# A_obs
# torch.Size([1, 8, 3, 3])
# V_tr
# torch.Size([1, 12, 3, 2])
# A_tr
# torch.Size([1, 12, 3, 3])


# dset_val = TrajectoryDataset(
#         data_set+'val/',
#         obs_len=obs_seq_len,
#         pred_len=pred_seq_len,
#         skip=1,norm_lap_matr=True)

# loader_val = DataLoader(
#         dset_val,
#         batch_size=1, #This is irrelative to the args batch size parameter
#         shuffle =False,
#         num_workers=1)


# #Defining the model 

# model = social_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
# output_feat=args.output_size,seq_len=args.obs_seq_len,
# kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()


# #Training settings 

# optimizer = optim.SGD(model.parameters(),lr=args.lr)

# if args.use_lrschd:
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)
    


# checkpoint_dir = './checkpoint/'+args.tag+'/'

# if not os.path.exists(checkpoint_dir):
#     os.makedirs(checkpoint_dir)
    
# with open(checkpoint_dir+'args.pkl', 'wb') as fp:
#     pickle.dump(args, fp)
    


# print('Data and model loaded')
# print('Checkpoint dir:', checkpoint_dir)

# #Training 
# metrics = {'train_loss':[],  'val_loss':[]}
# constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}
