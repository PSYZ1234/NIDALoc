import torch.nn as nn
import torch
import torch.nn.parallel
import torch.nn.functional as F
from utils.pointnet_util import PointNetSetAbstraction
from utils.cell_util import PCEncoding, HDCEncoding


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.sa1 = PointNetSetAbstraction(512,  4,    32,   3,       [32, 32, 64],     False, False)
        self.sa2 = PointNetSetAbstraction(128,  8,    16,   64 + 3,  [64, 128, 256],   False, False)
        self.sa3 = PointNetSetAbstraction(None, None, None, 256 + 3, [256, 512, 1024], False, True)

    def forward(self, xyz):
        B                 = xyz.size(0)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l3_points         = l3_points.view(B, -1) 

        return l3_points


class Decoder(nn.Module):
    def __init__(self, in_channel, mlp):
        super(Decoder, self).__init__()
        self.mlp_fcs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_fcs.append(nn.Linear(last_channel, out_channel))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, x):
        for i, fc in enumerate(self.mlp_fcs):
            bn = self.mlp_bns[i]
            x  = F.relu(bn(fc(x)))  
        
        return x


class NIDALoc(nn.Module):
    def __init__(self):
        super(NIDALoc, self).__init__()
        self.encoder        = Encoder()
        self.regressor1     = Decoder(1024, [1024, 1024, 1024])
        self.regressor2     = Decoder(1024, [1024])
        self.place_cell     = PCEncoding()
        self.hd_cell        = HDCEncoding()
        self.fc_position    = nn.Linear(1024, 3)
        self.fc_orientation = nn.Linear(1024, 3)
        self.fc_grid        = nn.Linear(1024, 2)


    def forward(self, x):
        y            = self.encoder(x)    
        pc           = self.place_cell(y)  
        z            = y + pc  
        # z            = y
        f_hd, cls_hd = self.hd_cell(z)    
        r0           = self.regressor1(z)  
        f_hd_norm    = F.normalize(f_hd, dim=-1)       
        r1           = r0 * f_hd_norm        
        # r1           = r0 
        r2           = self.regressor2(r1)
        t            = self.fc_position(r2)  
        q            = self.fc_orientation(r2)  
        grid         = self.fc_grid(r2)  
            
        return t, q, grid, cls_hd


if __name__ == '__main__':
    model = NIDALoc(8, 4096, 3)