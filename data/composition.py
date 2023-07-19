import torch
from torch.utils import data
from torch.autograd import Variable
import numpy as np
import sys
sys.path.insert(0, '../')
from .NCLT_datagenerator import NCLT
from .OxfordVelodyne_datagenerator import RobotCar
from utils.pose_util import calc_vos_simple, calc_vos_safe


class MF(data.Dataset):
    def __init__(self, dataset, include_vos=False, no_duplicates=False, *args, **kwargs):

        self.steps = kwargs.pop('steps', 2)
        self.skip = kwargs.pop('skip', 1)
        self.variable_skip = kwargs.pop('variable_skip', False)
        self.real = kwargs.pop('real', False)
        self.include_vos = include_vos
        self.train = kwargs['train']
        self.vo_func = kwargs.pop('vo_func', calc_vos_simple)
        # self.vo_func = kwargs.pop('vo_func', calc_vos_safe)
        self.no_duplicates = no_duplicates

        if dataset == 'NCLT':
            self.dset = NCLT(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = NCLT(*args, skip_pcs=True, real=False, **kwargs)
            self.threshold = 0.1
        elif dataset == 'Oxford':
            self.dset = RobotCar(*args, real=self.real, **kwargs)
            if self.include_vos and self.real:
                self.gt_dset = RobotCar(*args, skip_pcs=True, real=False, **kwargs)
            self.threshold = 0.02
        else:
            raise NotImplementedError('{:s} dataset is not implemented!')

        self.L = self.steps * self.skip

    def get_indices(self, index):
        if self.variable_skip:
            skips = np.random.randint(1, high=self.skip+1, size=self.steps-1)
        else:
            skips = self.skip * np.ones(self.steps-1)
        offsets = np.insert(skips, 0, 0).cumsum()  # (self.steps,)
        offsets -= offsets[len(offsets) // 2]
        if self.no_duplicates:
            offsets += np.ceil(self.steps/2 * self.skip)
        offsets = offsets.astype(np.int)
        idx = index + offsets
        idx = np.minimum(np.maximum(idx, 0), len(self.dset)-1)
        assert np.all(idx >= 0), '{:d}'.format(index)
        assert np.all(idx < len(self.dset))
        return idx

    def __getitem__(self, index):
        idx   = self.get_indices(index)
        clip  = [self.dset[i] for i in idx]
        pcs   = np.stack([c[0] for c in clip], axis=0)  # (self.steps, N, 3)
        poses = np.stack([c[1] for c in clip], axis=0)  # (self.steps, 6)
        grids = np.stack([c[2] for c in clip], axis=0)  # (self.steps, 2)
        hds   = np.stack([c[3] for c in clip], axis=0)  # (self.steps)

        if self.include_vos:
            vos = self.vo_func(poses[np.newaxis, ...])[0]
            if self.real:  # absolute poses need to come from the GT dataset
                clip = [self.gt_dset[self.dset.gt_idx[i]] for i in idx]
                poses = np.stack([c[1] for c in clip], axis=0)
            poses = np.concatenate((poses, vos), axis=0)

        return pcs, poses, grids, hds

    def __len__(self):
        L = len(self.dset)
        if self.no_duplicates:
            L -= (self.steps-1)*self.skip
        return L