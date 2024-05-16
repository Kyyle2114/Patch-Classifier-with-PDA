# Code Reference : https://github.com/ChuHan89/WSSS-Tissue/blob/main/tool/ADL_module.py

import torch
from torch import nn
import torch.nn.functional as F

class PDA(nn.Module):
    def __init__(self):
        """
        Progressive Dropout Attention class
        """
        super(PDA, self).__init__()
        
    def forward(self, 
                x: torch.Tensor, 
                fc_weights: torch.Tensor, 
                mu: float) -> torch.Tensor:
        
        fc_weights = fc_weights.view(1, -1, 1, 1)
        cams = x * fc_weights
        cams = F.relu(cams)
        
        N, C, _, _ = cams.size()
        cam_mean = torch.mean(cams, dim=1)
        
        zero = torch.zeros_like(cam_mean)        
        mean_drop_cam = zero
        
        for c in range(C):
            sub_cam = cams[:, c, :, :]
            sub_cam_max = torch.max(sub_cam.view(N, -1), dim=-1)[0].view(N, 1, 1)
            thr = (sub_cam_max * mu)
            thr = thr.expand(sub_cam.shape)
            sub_cam_with_drop = torch.where(sub_cam > thr, zero, sub_cam)
            mean_drop_cam = mean_drop_cam + sub_cam_with_drop
        
        mean_drop_cam = torch.unsqueeze(mean_drop_cam, dim=1)
        x = x * mean_drop_cam
        
        return x