import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GLIBlock(nn.Module):
    def __init__(self, channels, ratio,gamma = 2, b = 1):
        super(GLIBlock, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        )
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)

        self.cfc = Parameter(torch.Tensor(channels, 2))
        self.cfc.data.fill_(0)

        self.bn = nn.BatchNorm2d(channels)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

        self.sigmoid = nn.Sigmoid()

    def _style_integration(self, t):
        z = t * self.cfc[None, :, :]  # B x C x 2 对应元素相乘
        z = torch.sum(z, dim=2)[:, :, None, None] # B x C x 1 x 1

        # z_hat = self.bn(z)
        g = self.sigmoid(z)

        return g

    def forward(self, x, eps=1e-5):
        b, c, h, w = x.shape # x.shape = torch.Size([8, 512, 14, 14])

        # 全局最大池化
        avg_x_fc = self.avg_pooling(x).view(b, c) # [8, 512]

        # 全局最大池化 + 全局平均池化 + 全局标准差池化(conv1d)
        avg_x_conv = self.avg_pooling(x) # [8, 512,1,1]

        # 特征提取层
        v_fc = self.fc_layers(avg_x_fc)# + self.fc_layers(max_x_fc)  + self.fc_layers(std_x_fc) # [8,512]
        v_conv = self.conv(avg_x_conv.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)# + self.conv(max_x_conv.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) + self.conv(std_x_conv.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # [8,512,1,1]

        v_x,v_y = v_fc.shape # 8,512
        v_fc = v_fc.view(v_x,v_y,1) # [8, 512, 1]
        v_conv = v_conv.view(v_x,v_y,1) # [8, 512, 1]

        # 通道平均池化
        v_sum = torch.cat((v_fc, v_conv), dim=2)
        v_sum = self._style_integration(v_sum)
        # print(v_sum.size()) # [2, 3, 2]
        # v_sum = (v_fc + v_conv)/2

        # v = self.sigmoid(v_sum) # [8, 512, 1, 1]
         # [8, 512, 14, 14]
        return x * v_sum

if __name__ == "__main__":
    model = GLIBlock(512, 16,gamma = 2, b = 1)
    feature_maps1 = torch.randn((2, 3, 2, 2))
    feature_maps = torch.randn((8, 512, 14, 14))
    a= model(feature_maps)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
 
        self.h = h
        self.w = w
 
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
 
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
 
        return out
    

 #CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
 
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
 
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
 
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
 
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class GAM_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, rate=4):
        super(GAM_Attention, self).__init__()
 
        self.channel_attention = nn.Sequential(
            nn.Linear(in_channels, int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels / rate), in_channels)
        )
 
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), out_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(out_channels)
        )
 
    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
 
        x = x * x_channel_att
 
        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att #.expand_as(x)
 
        return out
 
#NAM
class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
 
 
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x) * residual #
        
        return x
 
 
class Att(nn.Module): #Yichao Liu, 2 months ago: • Add files via upload
    def __init__(self, channels,shape, out_channels=None, no_spatial=True):
        super(Att, self).__init__()
        self.Channel_Att = Channel_Att(channels)
  
    def forward(self, x):
        x_out1=self.Channel_Att(x)
 
        return x_out1  