# refer to https://github.com/open-mmlab/mmskeleton.git
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

#优雅操作张量维度
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# This is the original implementation for ST-GCN papers
# The based unit of graph convolutional networks.

class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """

    def __init__(self, in_channels, out_channels, s_kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1,
                 bias=True):
        super().__init__()

        self.s_kernel_size = s_kernel_size
        self.conv = nn.Conv2d(in_channels,
                              out_channels * s_kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)
        self.in_channels=in_channels
        self.out_channels=out_channels
    def forward(self, x, A):
        assert A.size(0) == self.s_kernel_size

        x = self.conv(x)
        # x=RandomSampler()
        n, kc, t, v = x.size()
        x = x.view(n, self.s_kernel_size, kc // self.s_kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        # #加入通道拓扑细化
        # x1 = CTRGC(self.in_channels, self.out_channels)
        # x = x+x1
        return x.contiguous(), A


class HD_Gconv(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, residual=True, att=False, CoM=21):
        super(HD_Gconv, self).__init__()
        self.num_layers = 1 #A.shape[0]  # 6
        self.num_subset = 3#A.size[0]  # 3

        self.att = att

        inter_channels = out_channels // (self.num_subset + 1)
        self.adaptive = adaptive

        if adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.numpy().astype(np.float32)), requires_grad=True)  # L, 3, 25, 25
            # self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            raise ValueError()

        self.conv_down = nn.ModuleList()
        self.conv = nn.ModuleList()
        for i in range(self.num_layers):
            self.conv_d = nn.ModuleList()
            self.conv_down.append(nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True)
            ))
            for j in range(self.num_subset):
                self.conv_d.append(nn.Sequential(
                    nn.Conv2d(inter_channels, inter_channels, kernel_size=1),
                    nn.BatchNorm2d(inter_channels)
                ))

            self.conv_d.append(EdgeConv(inter_channels, inter_channels, k=5))
            self.conv.append(self.conv_d)

        if self.att:
            self.aha = AHA(out_channels, num_layers=self.num_layers, CoM=CoM)

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0

        self.bn = nn.BatchNorm2d(out_channels)

        # 7개 conv layer
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x,A):

        # A = self.PA
        # assert A.size(0) == self.s_kernel_size

        out = []
        for i in range(self.num_layers):
            y = []
            x_down = self.conv_down[i](x)
            for j in range(self.num_subset):
                z = torch.einsum('n c t u, v u -> n c t v', x_down, A[j])
                z = self.conv[i][j](z)
                y.append(z)
            y_edge = self.conv[i][-1](x_down)
            y.append(y_edge)
            y = torch.cat(y, dim=1)  # N C T V

            out.append(y)

        # out = torch.tensor(out)
        out = torch.stack(out, dim=2)  # N C L T V
        # N,C,L,T,V = out.size()
        # out=out.reshape((N,C,T,V))

        # out = torch.squeeze(out, dim=1)  # 维度压缩，把1去掉

        if self.att:
            out = self.aha(out)  # N C T V
        else:
            out = out.sum(dim=2, keepdim=False)

        out = self.bn(out)

        out += self.down(x)
        out = self.relu(out)

        return out


class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(EdgeConv, self).__init__()

        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True, negative_slope=0.2)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, dim=4):  # N, C, T, V

        if dim == 3:
            N, C, L = x.size()
            pass
        else:
            N, C, T, V = x.size()
            x = x.mean(dim=-2, keepdim=False)  # N, C, V

        x = self.get_graph_feature(x, self.k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]

        if dim == 3:
            pass
        else:
            x = repeat(x, 'n c v -> n c t v', t=T)

        return x

    def knn(self, x, k):

        inner = -2 * torch.matmul(x.transpose(2, 1), x)  # N, V, V
        xx = torch.sum(x ** 2, dim=1, keepdim=True)
        pairwise_distance = - xx - inner - xx.transpose(2, 1)

        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # N, V, k
        return idx

    def get_graph_feature(self, x, k, idx=None):
        N, C, V = x.size()
        if idx is None:
            idx = self.knn(x, k=k)
        # 配置参数：
        # x.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # data = data.to(config.device)
        device = x.device #x.get_device()

        idx_base = torch.arange(0, N, device=device).view(-1, 1, 1) * V

        idx = idx + idx_base
        idx = idx.view(-1)

        x = rearrange(x, 'n c v -> n v c')
        feature = rearrange(x, 'n v c -> (n v) c')[idx, :]
        feature = feature.view(N, V, k, C)
        x = repeat(x, 'n v c -> n v k c', k=k)

        feature = torch.cat((feature - x, x), dim=3)
        feature = rearrange(feature, 'n v k c -> n c v k')

        return feature

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def get_groups(dataset='NTU', CoM=21):
    groups = []

    if dataset == 'NTU':
        if CoM == 2:
            groups.append([2])
            groups.append([1, 21])
            groups.append([13, 17, 3, 5, 9])
            groups.append([14, 18, 4, 6, 10])
            groups.append([15, 19, 7, 11])
            groups.append([16, 20, 8, 12])
            groups.append([22, 23, 24, 25])

        ## Center of mass : 21
        elif CoM == 21:
            groups.append([21])
            groups.append([2, 3, 5, 9])
            groups.append([4, 6, 10, 1])
            groups.append([7, 11, 13, 17])
            groups.append([8, 12, 14, 18])
            groups.append([22, 23, 24, 25, 15, 19])
            groups.append([16, 20])

        ## Center of Mass : 1
        elif CoM == 1:
            groups.append([1])
            groups.append([2, 13, 17])
            groups.append([14, 18, 21])
            groups.append([3, 5, 9, 15, 19])
            groups.append([4, 6, 10, 16, 20])
            groups.append([7, 11])
            groups.append([8, 12, 22, 23, 24, 25])

        else:
            raise ValueError()

    return groups