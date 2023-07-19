# pylint: disable=no-member
import argparse
import os
import sys
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.OxfordVelodyne_datagenerator import RobotCar
from data.NCLT_datagenerator import NCLT
from data.composition import MF
from data.augment import get_augmentations_from_list, Normalize
from models.model import NIDALoc
from models.loss import Criterion
from utils.pose_util import val_translation, val_rotation, val_classification, qexp
from tensorboardX import SummaryWriter
from torch.backends import cudnn
from torch.utils.data import DataLoader
from os import path as osp
from torchstat import stat
from thop import profile
from ptflops import get_model_complexity_info


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True

parser = argparse.ArgumentParser()
parser.add_argument('--multi_gpus', action='store_true', default=False, 
                    help='if use multi_gpus, default false')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu id for network, only effective when multi_gpus is false')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch Size during training [default: 80]')
parser.add_argument('--val_batch_size', type=int, default=32,
                    help='Batch Size during validating [default: 80]')
parser.add_argument('--max_epoch', type=int, default=99,
                    help='Epoch to run [default: 100]')
parser.add_argument('--init_learning_rate', type=float, default=0.001, 
                    help='Initial learning rate [default: 0.001]')
parser.add_argument("--decay_step", type=float, default=500,
                    help="decay step for learning rate, default: 1000")
parser.add_argument('--optimizer', default='adam',
                    help='adam or momentum [default: adam]')
parser.add_argument('--seed', type=int, default=20, metavar='S',
                    help='random seed (default: 20)')
parser.add_argument('--log_dir', default='NIDALoc-NCLT/',
                    help='Log dir [default: log]')
parser.add_argument('--dataset_folder', default='/home/data',
                    help='Our Dataset Folder')
parser.add_argument('--dataset', default='NCLT', 
                    help='Oxford or NCLT')
parser.add_argument('--num_workers', type=int, default=4, 
                    help='num workers for dataloader, default:4')
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points to downsample model to')
parser.add_argument('--augmentation', type=str, nargs='+', default=[],
                    choices=['Jitter', 'RotateSmall', 'Scale', 'Shift', 'Rotate1D', 'Rotate3D'],
                    help='Data augmentation settings to use during training')
parser.add_argument('--upright_axis', type=int, default=2,
                    help='Will learn invariance along this axis')
parser.add_argument('--num_loc', type=int, default=10, 
                    help='position classification, default: 10')
parser.add_argument('--num_ang', type=int, default=8, 
                    help='orientation classification, default: 10')
parser.add_argument('--skip', type=int, default=10, 
                    help='Number of frames to skip')
parser.add_argument('--steps', type=int, default=5, 
                    help='Number of frames to return on every call')
parser.add_argument('--normalize', action='store_true', default=False,
                    help='use normalize or not, default not')
parser.add_argument('--real', action='store_true', default=False, 
                    help='if True, load poses from SLAM / integration of VO')
parser.add_argument('--variable_skip', action='store_true', default=False, 
                    help='If True, skip = [1, ..., skip]')
parser.add_argument('--skip_val', action='store_true', default=False,
                    help='if skip validation during training, default False')
parser.add_argument('--resume_model', type=str, default='',
                    help='If present, restore checkpoint and resume training')
