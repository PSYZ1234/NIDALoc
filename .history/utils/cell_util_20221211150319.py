from os import X_OK
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F


class PCEncoding(nn.Module):
    def __init__(self):
        super(PCEncoding, self).__init__()
        self.step  = 5
        self.alpha = 0.01
        # self.alpha = 0.1
        # self.alpha = 1
        self.fc1   = nn.Linear(1024, 1024)
        self.bn1   = nn.BatchNorm1d(1024)

    def forward(self, x):  # [B*T, D]
        B     = x.size(0) // self.step
        x_seq = x.view(B, self.step, -1)  # [B, T, D] 
        # wirte
        h_0   = torch.zeros(B, 1024, 1024).cuda()
        memory_total = []
        for i in range(self.step):   
            k    = torch.unsqueeze(x_seq[:, i, :], 2)  # [B, D, 1] 
            v    = torch.unsqueeze(x_seq[:, i, :], 1)  # [B, 1, D] 
            memory_matrix = h_0    # [B, D, D] 
            hebb = self.alpha * (k * v - k**2 * memory_matrix)   # IKPCA2
            memory_matrix = hebb + memory_matrix  # [B, D, D] 
            h_0  = memory_matrix  # [B, D, D] 
            memory_total.append(memory_matrix)  # [B, D, D] 

        # read
        memory_total = torch.cat(memory_total, dim=0)  # [B*T, D, D]
        q            = memory_total @ x.view(B * self.step, -1, 1)  # [B*T, D, 1] 
        q            = q.view(B * self.step, -1)  # [B*T, D]
        out          = F.relu(self.bn1(self.fc1(q)))  # [B*T, D]
        out          = F.normalize(out) # new
        
        return out


# class PCEncoding(nn.Module):
#     def __init__(self):
#         super(PCEncoding, self).__init__()
#         self.step  = 5
#         self.alpha = 0.01 # original
#         # self.alpha = 0.1
#         # self.alpha = 1
#         self.fcin  = nn.Linear(1024, 512)
#         self.fc1   = nn.Linear(512, 1024)
#         self.bn1   = nn.BatchNorm1d(1024)  

#     def forward(self, x):  # [B*T, D]
#         x     = self.fcin(x)  # [B*T, D]
#         B     = x.size(0) // self.step
#         x_seq = x.view(B, self.step, -1)  # [B, T, D] 
#         # wirte
#         # h_0   = torch.zeros(B, 1024, 1024).cuda()
#         h_0   = torch.zeros(B, 512, 512).cuda()
#         # h_0   = torch.zeros(B, 2048, 2048).cuda()
#         memory_total = []
#         for i in range(self.step):   
#             k    = torch.unsqueeze(x_seq[:, i, :], 2)  # [B, D, 1] 
#             v    = torch.unsqueeze(x_seq[:, i, :], 1)  # [B, 1, D] 
#             memory_matrix = h_0    # [B, D, D] 
#             hebb = self.alpha * (k * v - k**2 * memory_matrix)   # IKPCA2
#             memory_matrix = hebb + memory_matrix  # [B, D, D] 
#             h_0  = memory_matrix  # [B, D, D] 
#             memory_total.append(memory_matrix)  # [B, D, D] 

#         # read
#         memory_total = torch.cat(memory_total, dim=0)  # [B*T, D, D]
#         q            = memory_total @ x.view(B * self.step, -1, 1)  # [B*T, D, 1] 
#         q            = q.view(B * self.step, -1)  # [B*T, D]
#         out          = F.relu(self.bn1(self.fc1(q)))  # [B*T, D]
#         out          = F.normalize(out) # new
        
#         return out



class HDCEncoding(nn.Module):
    def __init__(self):
        super(HDCEncoding, self).__init__()
        self.fc1  = nn.Linear(1024, 1024)
        self.bn1  = nn.BatchNorm1d(1024)
        self.fc2  = nn.Linear(1024, 1024)
        self.bn2  = nn.BatchNorm1d(1024)
        self.fc3  = nn.Linear(1024, 2)

    def forward(self, x):
        f   = F.relu(self.bn1(self.fc1(x)))  # [B*T, D]
        f   = F.relu(self.bn2(self.fc2(f)))  # [B*T, D]
        out = self.fc3(f)  # [B*T, D]
        out = F.log_softmax(out, dim=1)  # [B*T, D]

        return f, out