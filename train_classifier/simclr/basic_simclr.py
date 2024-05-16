import torch
import torch.nn as nn

class BasicSimCLR(nn.Module):
    def __init__(self, model, in_features, num_classes=128):
        """
        Basic SimCLR model 

        Args:
            model (nn.Module): pytorch model 
            in_features(int): input dimension of linear layer 
            num_classes (int, optional): dimension of z. Defaults to 128.
        """

        super(BasicSimCLR, self).__init__()
        
        self.backbone = model.backbone
        self.in_features = in_features
        
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(self.in_features, 512),
                                nn.ReLU(),
                                nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.avg(x)
        output = self.fc(x)
        return output