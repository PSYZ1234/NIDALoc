import torch
from torch import nn
import torch.nn.functional as F


class Criterion(nn.Module):
    def __init__(self):
        super(Criterion, self).__init__()
        self.t_loss_fn        = nn.L1Loss()
        self.q_loss_fn        = nn.L1Loss()
        self.grid_loss_fn     = nn.L1Loss()
        self.hd_loss_fn       = nn.NLLLoss()
        self.boundary_loss_fn = nn.NLLLoss()

    def forward(self, pred_t, pred_q, pred_grid, pred_hd, gt_t, gt_q, gt_grid, gt_hd):
        loss_pose     = 1 * self.t_loss_fn(pred_t, gt_t) + 10 * self.q_loss_fn(pred_q, gt_q)
        loss_grid     = self.grid_loss_fn(pred_grid, gt_grid) 
        loss_hd       = self.hd_loss_fn(pred_hd, gt_hd.long())
        # loss          = 1 * loss_pose + 0 * loss_grid + 5 * loss_hd
        # loss          = 1 * loss_pose + 5 * loss_grid + 0 * loss_hd
        # loss          = 1 * loss_pose + 0 * loss_grid + 0 * loss_hd
        loss          = 1 * loss_pose + 5 * loss_grid + 5 * loss_hd
        
        return loss


class Criterionlr(nn.Module):
    def __init__(self, sat=-3.0, saq=-3.0, learn_gamma=True):
        super(Criterionlr, self).__init__()
        self.t_loss_fn = nn.L1Loss()
        self.q_loss_fn = nn.L1Loss()
        self.sat = nn.Parameter(torch.tensor([sat], requires_grad=learn_gamma))
        self.saq = nn.Parameter(torch.tensor([saq], requires_grad=learn_gamma))

    def forward(self, pred_t, pred_q, gt_t, gt_q):
        loss_t = torch.exp(-self.sat) * self.t_loss_fn(pred_t, gt_t) + self.sat 
        loss_q = torch.exp(-self.saq) * self.q_loss_fn(pred_q, gt_q) + self.saq 
        loss   = loss_t + loss_q
        
        return loss