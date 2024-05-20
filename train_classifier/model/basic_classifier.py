import torch
import torch.nn as nn
import torchvision.models as models

from .dropout_attention import PDA

class BasicClassifier(nn.Module):
    def __init__(self, 
                 model, 
                 in_features, 
                 freezing=False, 
                 enable_PDA = False,
                 num_classes=1):
        """
        Basic Classifier with Global Average Pooling
        Add enable_PDA option 
        
        Args:
            model (nn.Module): pytorch model
            in_features(int): input dimension of linear layer 
            freezing (bool, optional): if True, freeze weight of backbone. Defaults to False.
            enable_PDA (bool, optional): if True, use Progressive Attention Dropout. Defaults to False.
            num_classes (int, optional): number of classes. Defaults to 1(binary classification).
        """
        super(BasicClassifier, self).__init__()
        
        self.enable_PDA = enable_PDA
        self.backbone = model.backbone
        
        if freezing:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )                   

        # Progressive Dropout Attention
        self.cnt = 0
        self.mu = 1.0 
        self.sigma = 0.985
        self.lb = 0.65
        self.PDA = PDA()


    def forward(self, x):
        x = self.backbone(x)
        
        if self.enable_PDA:
            x = self.PDA(x, self.fc[-1].weight, self.mu)
        
        x = self.avg(x)
        output = self.fc(x)
        
        return output
    
    def update_cutoff_(self) -> None:
        """
        Update PDA's cutoff(mu)
        """
        self.cnt += 1
        
        if self.cnt <= 3:
            self.mu = 1.0
        
        if self.cnt >= 4:
            if self.mu > self.lb:
                self.mu = self.mu * self.sigma