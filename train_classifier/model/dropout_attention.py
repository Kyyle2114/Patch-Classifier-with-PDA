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
        
        # K: total number of category(class) 
        # N: batch size 
        K, _ = fc_weights.size()
        N, _, H, W = x.size()
        
        zero = torch.zeros(size=(N, H, W)).to(x.device)
        mean_drop_cam = zero
        
        for k in range(K):
            fc_weight = fc_weights[k].view(1, -1, 1, 1)
            cam = x * fc_weight
            cam = F.relu(cam)
            cam = torch.sum(cam, dim=1)
            
            cam_max = torch.max(cam.view(N, -1), dim=-1)[0].view(N, 1, 1)
            thr = (cam_max * mu)
            thr = thr.expand(cam.shape)
            cam_with_drop = torch.where(cam > thr, zero, cam).to(x.device)
            
            mean_drop_cam = mean_drop_cam + cam_with_drop
        
        mean_drop_cam = mean_drop_cam / K 
        mean_drop_cam = mean_drop_cam.unsqueeze(dim=1)
        
        x = x * mean_drop_cam
        
        return x