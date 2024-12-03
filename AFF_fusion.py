import torch
import math
from torch import nn
# from Fusionlist.SeBlock import *
# from EcaBlock import *


class AFF(nn.Module):
    def __init__(self, channels=512,r=4):
        super(AFF, self).__init__()
        inter_channels = int(channels // r) 
        # kernel_size = int(abs((math.log(channels, 2) + 1) / 2))
        # print(kernel_size)
        # print((kernel_size - 1) // 2)
        self.conv1 = nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.local_att1 = nn.Sequential(
            nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
        )

        # self.conv = nn.Conv1d(1, 1, kernel_size = 3, padding = 1, bias = False)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(channels)

        # self.local_att2 = nn.Sequential(
        #     nn.Conv1d(1, 1, kernel_size = 3, padding =1, bias = False),
        #     nn.BatchNorm2d(channels),
        # )
        self.local_att2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3, 3), stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )


        # self.attention1 = ECABlock(channels, gamma = 2, b = 1)
 
        self.attention2 = SEBlock(channels, 16)
 
        self.sigmoid = nn.Sigmoid()
 
 
    def forward(self, x, residual):
        xa = x + residual

        xz1 = self.local_att1(xa)
        # xg1 = self.attention1(xz1)
        

        # xz2 = self.conv(xa.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        xz2 = self.conv(xa)
        
        xz2 = self.bn(xz2)
        # xz2 = self.bn(xz2)
        # xg2 = self.attention2(xz2)

        xlg1 = xz2 + xz1
        xlg1 = self.attention2(xlg1)
        wei = self.sigmoid(xlg1)
 
 
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        return xo
    

class SEBlock(nn.Module):
    def __init__(self, channels, ratio):
        super(SEBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        # self.max_pooling = nn.AdaptiveMaxPool2d(1)
        # if mode == "max":
        #     self.global_pooling = self.max_pooling
        # elif mode == "avg":
        #     self.global_pooling = self.avg_pooling
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False),
        )
        self.sigmoid = nn.Sigmoid()
     
    
    def forward(self, x):
        b, c, _, _ = x.shape
        v = self.avg_pooling(x).view(b, c)
        # v = x.view(b,c)
        v = self.fc_layers(v).view(b, c, 1, 1)
        v = self.sigmoid(v)
        return x*v    
    
if __name__ == "__main__":

    model = AFF(channels)#.cuda()
    print("Model loaded.")
    x1 = torch.rand(2, 512,1,1)#.cuda()
    x2 = torch.rand(2, 512,1,1)#.cuda()
    print("x1 and x2 loaded.")

	# Run a feedforward and check shape
    c = model(x1, x2)
    print(image.shape)
    print(c.shape)