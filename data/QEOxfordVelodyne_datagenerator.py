import os
import torch
import numpy as np
import pickle
import os.path as osp
import h5py
import json
from data.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses, interpolate_vo_poses
from data.robotcar_sdk.python.transform import build_se3_transform
from data.robotcar_sdk.python.velodyne import load_velodyne_binary
from torch.utils import data
from utils.pose_util import process_poses, ds_pc, filter_overflow_ts
from utils.pose_util import grid_position, hd_orientation
from copy import deepcopy
import transforms3d.quaternions as txq


BASE_DIR = osp.dirname(osp.abspath(__file__))

def calibrate_process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    poses_out = np.zeros((len(poses_in), 6))
    rot_out = np.zeros((len(poses_in), 3, 3))
    poses_out[:, 0:3] = poses_in[:, 9:]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i, :9].reshape((3, 3))
        rot_out[i, :, :] = R
        q = txq.mat2quat(np.dot(align_R, R))
        q *= np.sign(q[0])  # constrain to hemisphere
        q = qlog(q)
        poses_out[i, 3:] = q
        t = poses_out[i, :3] - align_t
        poses_out[i, :3] = align_s * np.dot(align_R, t[:, np.newaxis]).squeeze()

    # normalize translation
    poses_out[:, :3] -= mean_t
    poses_out[:, :3] /= std_t

    # max and min
    pose_max = np.max(poses_out[:, :2], axis=0)  # (2,)
    pose_min = np.min(poses_out[:, :2], axis=0)  # (2,)

    return poses_out, rot_out, pose_max, pose_min


def qlog(q):
    """
    Applies logarithm map to q
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])

    return q


class RobotCar(data.Dataset):
    def __init__(self, data_path, train=True, valid=False, augmentation=[], num_points=4096, real=False,
                 vo_lib='stereo', num_loc=10, num_ang=10):
        # directories
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'Oxford')
        extrinsics_dir = osp.join(BASE_DIR, 'robotcar_sdk', 'extrinsics')

        # decide which sequences to use
        if train:
            split_filename = osp.join(data_dir, 'train_split.txt')
            # split_filename = osp.join(data_dir, 'lw_train_split.txt')
        elif valid:
            split_filename = osp.join(data_dir, 'valid_split.txt')
            # split_filename = osp.join(data_dir, 'lw_valid_split.txt')
        else:
            split_filename = osp.join(data_dir, 'test_split.txt')
        with open(split_filename, 'r') as f:
            seqs = [l.rstrip() for l in f if not l.startswith('#')]

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        # extrinsic reading
        with open(os.path.join(extrinsics_dir, lidar + '.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
        G_posesource_laser = build_se3_transform([float(x) for x in extrinsics.split(' ')])
        with open(os.path.join(extrinsics_dir, 'ins.txt')) as extrinsics_file:
            extrinsics = next(extrinsics_file)
            G_posesource_laser = np.linalg.solve(build_se3_transform([float(x) for x in extrinsics.split(' ')]),
                                                 G_posesource_laser)  # (4, 4)
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            # read the image timestamps
            h5_path = osp.join(seq_dir, lidar + '_calibrate' + 'False.h5')
            if not os.path.isfile(h5_path):
                print('interpolate ' + seq)
                ts_filename = osp.join(seq_dir, lidar + '.timestamps')
                with open(ts_filename, 'r') as f:
                    ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
                ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
                ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
                rot = np.fromfile(osp.join(seq_dir, 'rot_tr.bin'), dtype=np.float32).reshape(-1, 9)
                t = np.fromfile(osp.join(seq_dir, 'tr_add_mean.bin'), dtype=np.float32).reshape(-1, 3)
                ps[seq] = np.concatenate((rot, t), axis=1)  # (n, 12)
                # write to h5 file
                print('write interpolate pose to ' + h5_path)
                h5_file = h5py.File(h5_path, 'w')
                h5_file.create_dataset('valid_timestamps', data=np.asarray(ts[seq], dtype=np.int64))
                h5_file.create_dataset('poses', data=ps[seq])
            else:
                # load h5 file, save pose interpolating time
                print("load " + seq + ' pose from ' + h5_path)
                h5_file = h5py.File(h5_path, 'r')
                ts[seq] = h5_file['valid_timestamps'][...]
                ps[seq] = h5_file['poses'][...]

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # read / save pose normalization information
        poses = np.empty((0, 12))
        for p in ps.values():
            poses = np.vstack((poses, p))
        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        if train:
            mean_t = np.mean(poses[:, 9:], axis=0)  # (3,)
            std_t = np.std(poses[:, 9:], axis=0)  # (3,)
            np.savetxt(pose_stats_filename, np.vstack((mean_t, std_t)), fmt='%8.7f')
        else:
            mean_t, std_t = np.loadtxt(pose_stats_filename)

        # convert the pose to translation + log quaternion, align, normalize
        self.poses     = np.empty((0, 6))
        self.poses_max = np.empty((0, 2))
        self.poses_min = np.empty((0, 2))   
        pose_max_min_filename = osp.join(data_dir, 'pose_max_min.txt')

        for seq in seqs:
            pss, _, pss_max, pss_min = calibrate_process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                                                                  align_R=vo_stats[seq]['R'],
                                                                  align_t=vo_stats[seq]['t'],
                                                                  align_s=vo_stats[seq]['s'])
            self.poses     = np.vstack((self.poses, pss)) 
            self.poses_max = np.vstack((self.poses_max, pss_max))
            self.poses_min = np.vstack((self.poses_min, pss_min)) 
            
        if train:
            self.poses_max = np.max(self.poses_max, axis=0)  # (2,)
            self.poses_min = np.min(self.poses_min, axis=0)  # (2,)
            center_point   = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2) 
            np.savetxt(pose_max_min_filename, np.vstack((self.poses_max, self.poses_min)), fmt='%8.7f')
        else:
            self.poses_max, self.poses_min = np.loadtxt(pose_max_min_filename)
            center_point = list((np.array(list(self.poses_min)) + np.array(list(self.poses_max))) / 2) 

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
        ptcld     = load_velodyne_binary(scan_path)  # (4, N)
        scan      = ptcld[:3].transpose()  # (N, 3)
        scan      = ds_pc(scan, self.num_points)
        for a in self.augmentation:
            scan  = a.apply(scan) 
            
        pose      = self.poses[index]  # (6,)  all-[156988, 6]
        grid      = grid_position(pose, self.poses_max, self.poses_min, self.num_loc) # (2, )
        hd        = hd_orientation(pose, self.num_ang)  # (1, )

        return scan, pose, grid, hd

    def __len__(self):
        return len(self.poses)


if __name__ == '__main__':
    velodyne_dataset = RobotCar(data_path='/home/yss/Data/Oxford', train=True, valid=True)
    print("finished")