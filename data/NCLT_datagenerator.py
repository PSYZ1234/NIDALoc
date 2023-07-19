import os
import os.path as osp
import h5py
import scipy
import numpy as np
from scipy.interpolate import interp1d
from utils.pose_util import *
from numpy import mean
from copy import deepcopy
from torch.utils import data
from torch.utils.data import DataLoader
from data.robotcar_sdk.python.transform import build_se3_transform
from utils.pose_util import process_poses, ds_pc
from utils.pose_util import grid_position, hd_orientation

BASE_DIR = osp.dirname(osp.abspath(__file__))


def filter_overflow_nclt(gt_filename, ts_raw): # 滤波函数
    # gt_filename: GT对应的文件名
    # ts_raw: 原始数据集提供的点云时间戳
    ground_truth        = np.loadtxt(gt_filename, delimiter=",")[1:,0]
    min_pose_timestamps = min(ground_truth)
    max_pose_timestamps = max(ground_truth)
    ts_filted           = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num         = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, gt_filename))

    return ts_filted


def interpolate_pose_nclt(gt_filename, ts_raw): # 插值函数
    # gt_filename: GT对应文件名
    # ts_raw: 滤波后的点云时间戳
    ground_truth = np.loadtxt(gt_filename, delimiter=",")
    interp       = scipy.interpolate.interp1d(ground_truth[1:, 0], ground_truth[1:, 1:], kind='nearest', axis=0)
    pose_gt      = interp(ts_raw)
    print(pose_gt.shape)

    return pose_gt


def so3_to_euler_nclt(poses_in):
    N         = len(poses_in)
    poses_out = np.zeros((N, 4, 4))
    for i in range(N):
        poses_out[i, :, :] = build_se3_transform([poses_in[i, 0], poses_in[i, 1], poses_in[i, 2],
                                                  poses_in[i, 3], poses_in[i, 4], poses_in[i, 5]])

    return poses_out


class NCLT(data.Dataset):
    def __init__(self, data_path, train=True, valid=False, augmentation=[], num_points=4096, real=False, 
                 vo_lib='stereo', num_loc=10, num_ang=10):
        # data_path    : 数据地址
        # train valid  : 指示导入训练或者测试数据
        # augmentation : 指示数据增强方式
        # real         : 指示是否采用插值

        lidar    = 'velodyne_left'
        data_dir = osp.join(data_path, 'NCLT')
        # extrinsics_dir = osp.join(BASE_DIR, 'robotcar_sdk', 'extrinsics')

        # decide which sequences to use
        if train:
            split_filename = osp.join(data_dir, 'train_split.txt')
        elif valid:
            split_filename = osp.join(data_dir, 'valid_split.txt')
        else:
            split_filename = osp.join(data_dir, 'test_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        # extrinsic reading
        # with open(os.path.join(extrinsics_dir, 'nlct_velodyne' + '.txt')) as extrinsics_file:
        #     extrinsics = next(extrinsics_file)
        # G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
        # with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
        #     extrinsics = next(extrinsics_file)
        #     G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
        #                                          G_posesource_laser)  # (4, 4)
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)
            h5_path = osp.join(seq_dir, lidar + '_' + str(real) + '.h5')
            # 如果不存在上述h5文件，则创建并保存
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_raw = []
                # 读入LiDAR时间戳，并从小到大排序
                vel = os.listdir(seq_dir + '/velodyne_left')
                for i in range(len(vel)):
                    ts_raw.append(int(vel[i][:-4]))
                ts_raw = sorted(ts_raw)
                # GT poses
                gt_filename = osp.join(seq_dir, 'groundtruth_' + seq + '.csv')
                ts[seq]     = filter_overflow_nclt(gt_filename, ts_raw)
                p           = interpolate_pose_nclt(gt_filename, ts[seq])  # (n, 6)
                p           = so3_to_euler_nclt(p)  # (n, 4, 4)
                # p = np.asarray([np.dot(pose, G_posesource_laser) for pose in p])  # (n, 4, 4)
                ps[seq]     = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)

                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file     = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file     = h5py.File(h5_path, 'r')
                ts[seq]     = h5_file['valid_timestamps'][...]
                ps[seq]     = h5_file['poses'][...]

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # 读取/ 记录位姿归一化信息
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train:
            mean_t = np.mean(poses[:, [3, 7, 11]], axis=0)  # (3,)
            std_t  = np.std(poses[:, [3, 7, 11]], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # 将pose转换为平移+对数四元数、对齐，归一化
        self.poses      =  np.empty((0, 6))
        self.poses_max  =  np.empty((0, 2))
        self.poses_min  =  np.empty((0, 2))
        pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')

        for seq in seqs:
            pss, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                  align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                  align_s=vo_stats[seq]['s'])
            self.poses     = np.vstack((self.poses, pss)) 
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min))

        if train:
            self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
            self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)

        self.augmentation = augmentation
        self.num_points   = num_points
        self.num_loc = num_loc
        self.num_ang = num_ang

        if train:
            print("train data num:" + str(len(self.poses)))
            print("train grid num:" + str(self.num_loc * self.num_loc))
        else:
            print("valid data num:" + str(len(self.poses)))
            print("valid grid num:" + str(self.num_loc * self.num_loc))

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        scan      = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)  # (N, 4)
        scan      = scan[:, :3]
        # 下采样到规定点
        scan = ds_pc(scan, self.num_points)
        for a in self.augmentation:
            scan = a.apply(scan)

        pose = self.poses[index]  # (6,)
        grid = grid_position(pose, self.poses_max, self.poses_min, self.num_loc) # (2, )
        hd   = hd_orientation(pose, self.num_ang)  # (1, )

        return scan, pose, grid, hd

    def __len__(self):
        return len(self.poses)