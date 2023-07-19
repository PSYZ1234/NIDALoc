import numpy as np
import math
import torch
import pandas as pd
import transforms3d.quaternions as txq
import transforms3d.euler as txe
from os import path as osp


def vdot(v1, v2):
    """
    Dot product along the dim=1
    :param v1: N x d
    :param v2: N x d
    :return: N x 1
    """
    out = torch.mul(v1, v2)
    out = torch.sum(out, 1)

    return out


def normalize(x, p=2, dim=0):
    """
    Divides a tensor along a certain dim by the Lp norm
    :param x:
    :param p: Lp norm
    :param dim: Dimension to normalize along
    :return:
    """
    xn = x.norm(p=p, dim=dim)
    x  = x / xn.unsqueeze(dim=dim)

    return x


def qmult(q1, q2):
    """
    Multiply 2 quaternions
    :param q1: Tensor N x 4
    :param q2: Tensor N x 4
    :return: quaternion product, Tensor N x 4
    """
    q1s, q1v = q1[:, :1], q1[:, 1:]
    q2s, q2v = q2[:, :1], q2[:, 1:]

    qs = q1s * q2s - vdot(q1v, q2v)
    qv = q1v.mul(q2s.expand_as(q1v)) + q2v.mul(q1s.expand_as(q2v)) + \
         torch.cross(q1v, q2v, dim=1)
    q  = torch.cat((qs, qv), dim=1)

    # normalize
    q  = normalize(q, dim=1)

    return q


def qinv(q):
    """
    Inverts quaternions
    :param q: N x 4
    :return: q*: N x 4
    """
    q_inv = torch.cat((q[:, :1], -q[:, 1:]), dim=1)

    return q_inv


def qlog(q):
    """
    Applies the log map to a quaternion
    :param q: (4,)
    :return: (3,)
    """
    if all(q[1:] == 0):
        q = np.zeros(3)
    else:
        q = np.arccos(q[0]) * q[1:] / np.linalg.norm(q[1:])
        
    return q


def qexp(q):
    """
    Applies exponential map to log quaternion
    :param q: (3,)
    :return: (4,)
    """
    n = np.linalg.norm(q)
    q = np.hstack((np.cos(n), np.sinc(n/np.pi)*q))

    return q


def qexp_t_safe(q):
    """
    Applies exponential map to log quaternion (safe implementation that does not
    maintain gradient flow)
    :param q: N x 3
    :return: N x 4
    """
    q = torch.from_numpy(np.asarray([qexp(qq) for qq in q.numpy()],
                                    dtype=np.float32))

    return q


def qlog_t_safe(q):
    """
    Applies the log map to a quaternion (safe implementation that does not
    maintain gradient flow)
    :param q: N x 4
    :return: N x 3
    """
    q = torch.from_numpy(np.asarray([qlog(qq) for qq in q.numpy()],
                                    dtype=np.float32))

    return q


def rotate_vec_by_q(t, q):
    """
    rotates vector t by quaternion q
    :param t: vector, Tensor N x 3
    :param q: quaternion, Tensor N x 4
    :return: t rotated by q: t' = t + 2*qs*(qv x t) + 2*qv x (qv x r)
    """
    qs, qv = q[:, :1], q[:, 1:]
    b      = torch.cross(qv, t, dim=1)
    c      = 2 * torch.cross(qv, b, dim=1)
    b      = 2 * b.mul(qs.expand_as(b))
    tq     = t + b + c

    return tq


def calc_vo_logq_safe(p0, p1):
    """
    VO in the p0 frame using numpy fns
    :param p0:
    :param p1:
    :return:
    """
    vos_t = p1[:, :3] - p0[:, :3]
    q0    = qexp_t_safe(p0[:, 3:])
    q1    = qexp_t_safe(p1[:, 3:])
    vos_t = rotate_vec_by_q(vos_t, qinv(q0))
    vos_q = qmult(qinv(q0), q1)
    vos_q = qlog_t_safe(vos_q)

    return torch.cat((vos_t, vos_q), dim=1)


def calc_vos_safe(poses):
    """
    calculate the VOs, from a list of consecutive poses
    :param poses: N x T x 6
    :return: N x (T-1) x 6
    """
    vos = []
    for p in poses:
        pvos = [calc_vo_logq_safe(p[i].unsqueeze(0), p[i + 1].unsqueeze(0))
                for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)
    return vos


def calc_vos_simple(poses):
    """
    calculate the VOs, from a list of consecutive poses
    :param poses: N x T x 6
    :return: N x (T-1) x 6
    """
    vos = []
    for p in poses:
        pvos = [p[i + 1].unsqueeze(0) - p[i].unsqueeze(0) for i in range(len(p) - 1)]
        vos.append(torch.cat(pvos, dim=0))
    vos = torch.stack(vos, dim=0)

    return vos


def process_poses(poses_in, mean_t, std_t, align_R, align_t, align_s):
    """
    processes the 1x12 raw pose from dataset by aligning and then normalizing
    :param poses_in: N x 12
    :param mean_t: 3
    :param std_t: 3
    :param align_R: 3 x 3
    :param align_t: 3
    :param align_s: 1
    :return: processed poses (translation + quaternion) N x 7
    """
    poses_out = np.zeros((len(poses_in), 6))
    poses_out[:, 0:3] = poses_in[:, [3, 7, 11]]

    # align
    for i in range(len(poses_out)):
        R = poses_in[i].reshape((3, 4))[:3, :3]
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

    return poses_out, pose_max, pose_min


def val_translation(pred_p, gt_p):
    """
    test model, compute error (numpy)
    input:
        pred_t: [B, 3]
        gt_t: [B, 3]
    returns:
        translation error (m):
    """
    if isinstance(pred_p, np.ndarray):
        predicted   = pred_p
        groundtruth = gt_p
    else:
        predicted   = pred_p.cpu().numpy()
        groundtruth = gt_p.cpu().numpy()
    error = np.linalg.norm(groundtruth - predicted)

    return error


def val_rotation(pred_q, gt_q):
    """
    test model, compute error (numpy)
    input:
        pred_q: [B, 3]
        gt_q: [B, 3]
    returns:
        rotation error (degrees):
    """
    if isinstance(pred_q, np.ndarray):
        predicted   = pred_q
        groundtruth = gt_q
    else:
        predicted   = pred_q.cpu().numpy()
        groundtruth = gt_q.cpu().numpy()

    d     = abs(np.dot(groundtruth, predicted))
    d     = min(1.0, max(-1.0, d))
    error = 2 * np.arccos(d) * 180 / np.pi

    return error


def val_classification(pred_cls, gt_cls):
    """
    test model, compute error (numpy)
    input:
        pred_cls: [B, D]
        gt_cls: [B]
    returns:
        correct_cls:
    """
    pred_choice = pred_cls.max(1)[1]
    correct_cls = pred_choice.eq(gt_cls.long()).cpu().sum()

    return correct_cls


def ds_pc(cloud, target_num):
    if cloud.shape[0] <= target_num:
        # Add in artificial points if necessary
        print('Only %i out of %i required points in raw point cloud. Duplicating...' % (cloud.shape[0], target_num))
        num_to_pad = target_num - cloud.shape[0]
        pad_points = cloud[np.random.choice(cloud.shape[0], size=num_to_pad, replace=True), :]
        cloud      = np.concatenate((cloud, pad_points), axis=0)

        return cloud
    else:
        cloud = cloud[np.random.choice(cloud.shape[0], size=target_num, replace=False), :]

        return cloud


def filter_overflow_ts(filename, ts_raw):
    file_data = pd.read_csv(filename)
    base_name = osp.basename(filename)
    if base_name.find('vo') > -1:
        ts_key = 'source_timestamp'
    else:
        ts_key = 'timestamp'
    pose_timestamps     = file_data[ts_key].values
    min_pose_timestamps = min(pose_timestamps)
    max_pose_timestamps = max(pose_timestamps)
    ts_filted           = [t for t in ts_raw if min_pose_timestamps < t < max_pose_timestamps]
    abandon_num         = len(ts_raw) - len(ts_filted)
    print('abandom %d pointclouds that exceed the range of %s' % (abandon_num, filename))
    
    return ts_filted


def grid_position(pose, pose_max, pose_min, num_loc):
    """
    get boundary position
    :param pose: [6,]
    :param pose_max: [2,]
    :param pose_min: [2,]
    :param num_grid: 10
    :return: boundary position [2,]
    """
    len_x      = pose_max[0] - pose_min[0]
    len_y      = pose_max[1] - pose_min[1]
    x          = (pose[0] - pose_min[0]) / (len_x)
    y          = (pose[1] - pose_min[1]) / (len_y)
    x          = np.maximum(x, 0)
    y          = np.maximum(y, 0)
    x          = int(np.minimum(x * num_loc, (num_loc - 1)))
    y          = int(np.minimum(y * num_loc, (num_loc - 1)))
    boundary_x = ((x + 1) / num_loc) *  len_x + pose_min[0] - len_x * (1 / (num_loc * 2))
    boundary_y = ((y + 1) / num_loc) *  len_y + pose_min[1] - len_y * (1 / (num_loc * 2))
    boundary_x = np.array([boundary_x])
    boundary_y = np.array([boundary_y])
    boundary   = np.concatenate((boundary_x, boundary_y), axis=0)

    return boundary


def hd_orientation(pose, num_ang):
    """
    :param pose: [6,]
    :param num_ang: 10
    :return: class k
    """
    quat    = qexp(pose[3:])
    z, y, x = txe.quat2euler(quat)
    # print(x)
    theta   = math.degrees(x)
    if theta<-180 or theta>180:
        raise ValueError("angle error!")

    cls_ori = (theta - math.degrees(-math.pi)) / (math.degrees(math.pi) - math.degrees(-math.pi))
    cls_ori = int(np.minimum(cls_ori * num_ang, (num_ang - 1)))
    
    if cls_ori == 0 or cls_ori == 2 or cls_ori == 4 or cls_ori == 6:
        class_orientation = 1
    else:
        class_orientation = 0

    # if cls_ori == 0 or cls_ori == 2 or cls_ori == 4 or cls_ori == 6 \
    # or cls_ori == 8 or cls_ori == 10 or cls_ori == 12 or cls_ori == 14:
    #     class_orientation = 1
    # else:
    #     class_orientation = 0

    # if cls_ori == 0 or cls_ori == 2 or cls_ori == 4 or cls_ori == 6 \
    # or cls_ori == 8 or cls_ori == 10 or cls_ori == 12 or cls_ori == 14 \
    # or cls_ori == 16 or cls_ori == 18 or cls_ori == 20 or cls_ori == 22 \
    # or cls_ori == 24 or cls_ori == 26 or cls_ori == 28 or cls_ori == 30:
    #     class_orientation = 1
    # else:
    #     class_orientation = 0

    return class_orientation