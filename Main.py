import json
import cv2
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import math

from pytorchtools import EarlyStopping
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt

import shutil
import time
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from torch.optim import *
from torchvision.transforms import *
from utils.mydata_xu import *
from ST_GCN.ST_GCN_Block import ST_GCN_18
# from Swin_TF.swin_transformer import SwinTransformer3D,SwinTransformerBlock3D
from Fusion import AFF as RDF
# from CMT import cmt_s
# 设置 max_split_size_mb 为256MB
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

EMOTION_LABEL = ['Anxiety', 'Peace', 'Weariness', 'Happiness', 'Anger']
DRIVER_BEHAVIOR_LABEL = ['Smoking', 'Making Phone', 'Looking Around', 'Dozing Off', 'Normal Driving', 'Talking',
                         'Body Movement']
SCENE_CENTRIC_CONTEXT_LABEL = ['Traffic Jam', 'Waiting', 'Smooth Traffic']
VEHICLE_BASED_CONTEXT_LABEL = ['Parking', 'Turning', 'Backward Moving', 'Changing Lane', 'Forward Moving']



class CarDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.path = pd.read_csv(csv_file)
        # self.path='/root/'+self.path
        self.transform = transform
        self.resize_height = 224
        self.resize_width = 224
        self.body_height = 112
        self.body_width = 112
        self.face_height = 64#56 #64
        self.face_width = 64#56 #64

    # /root/autodl-tmp/AIDE_Dataset/AIDE_Dataset/annotation/0006.json
    def __len__(self):
        return len(self.path)

    # 为数据加载器提供单个样本，包括图像数据和相关标签
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        frames_path, label_path = self.path.iloc[idx]
        frames_path = '/root/' + frames_path
        label_path = '/root/' + label_path

        parts1 = frames_path.split('/')
        # parts1.insert(4, 'AIDE_Dataset')  # 在第四个元素后添加 'AIDE_Dataset'
        frames_path = '/'.join(parts1)

        parts2 = label_path.split('/')
        # parts2.insert(4, 'AIDE_Dataset')  # 在第四个元素后添加 'AIDE_Dataset'
        label_path = '/'.join(parts2)

        label_json = json.load(open(label_path))
        pose_list = label_json['pose_list']

        # buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body,keypoints = self.load_frames(frames_path, pose_list)  # 加载图像帧数据
        buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture, gesture = self.load_frames(frames_path, pose_list)

        # buffer, buffer_front, buffer_left, buffer_right, buffer_body, buffer_face, keypoints = self.load_frames(
        #     frames_path, pose_list)  # 加载图像帧数据

        # 数据增强
        buffer = self.randomflip(buffer)
        buffer_front = self.randomflip(buffer_front)
        buffer_left = self.randomflip(buffer_left)
        buffer_right = self.randomflip(buffer_right)

        context = torch.cat([buffer, buffer_front, buffer_left, buffer_right], dim=0)  # 将四个张量沿批次维度拼接
        context = self.to_tensor(context)

        # 加载的图像数据-->PyTorch张量
        buffer = self.to_tensor(buffer)
        buffer_front = self.to_tensor(buffer_front)
        buffer_left = self.to_tensor(buffer_left)
        buffer_right = self.to_tensor(buffer_right)

        # 身体、面部、关节点
        buffer_body = self.to_tensor(buffer_body)
        buffer_face = self.to_tensor(buffer_face)
        # keypoints = keypoints.permute(2, 0, 1).contiguous()

        emotion_label = EMOTION_LABEL.index((label_json['emotion_label'].capitalize()))
        driver_behavior_label = DRIVER_BEHAVIOR_LABEL.index((label_json['driver_behavior_label']))
        scene_centric_context_label = SCENE_CENTRIC_CONTEXT_LABEL.index((label_json['scene_centric_context_label']))

        # 标签错误情况
        if label_json['vehicle_based_context_label'] == "Forward":
            label_json['vehicle_based_context_label'] = "Forward Moving"
        # print(label_json['vehicle_based_context_label'], label_path)
        vehicle_based_context_label = VEHICLE_BASED_CONTEXT_LABEL.index((label_json['vehicle_based_context_label']))

        sample = {
            'context': context,
            'body': buffer_body,
            'face': buffer_face,
            # 'keypoints': torch.stack([keypoints], dim=-1),
            'posture': posture,  # 使用 posture
            'gesture': gesture,  # 使用 gesture
            "emotion_label": emotion_label,
            "driver_behavior_label": driver_behavior_label,
            "scene_centric_context_label": scene_centric_context_label,
            "vehicle_based_context_label": vehicle_based_context_label
        }

        # keypoints = sample['keypoints']
        posture = sample['posture']  # 使用 posture
        gesture = sample['gesture']  # 使用 gesture
        context = sample['context']
        body = sample['body']
        face = sample['face']
        emotion_label = sample['emotion_label']
        behavior_label = sample['driver_behavior_label']
        context_label = sample['scene_centric_context_label']
        vehicle_label = sample['vehicle_based_context_label']
      

        # 返回图像数据和相关标签
        return buffer, buffer_front, buffer_left, buffer_right, buffer_face, buffer_body, posture,gesture, emotion_label, behavior_label, context_label, vehicle_label


    def load_frames(self, file_dir, pose_list):

        incar_path = os.path.join(file_dir, 'incarframes')
        front_frames = os.path.join(file_dir, 'frontframes')
        left_frames = os.path.join(file_dir, 'leftframes')
        right_frames = os.path.join(file_dir, 'rightframes')
        face_frames = os.path.join(file_dir, 'face')
        body_frames = os.path.join(file_dir, 'body')


        frames = [os.path.join(incar_path, img) for img in os.listdir(incar_path) if img.endswith('.jpg')]
        front_frames = [os.path.join(front_frames, img) for img in os.listdir(front_frames) if img.endswith('.jpg')]
        left_frames = [os.path.join(left_frames, img) for img in os.listdir(left_frames) if img.endswith('.jpg')]
        right_frames = [os.path.join(right_frames, img) for img in os.listdir(right_frames) if img.endswith('.jpg')]

        face_frames = [os.path.join(face_frames, img) for img in os.listdir(face_frames) if img.endswith('.jpg')]
        if len(face_frames)!=45:
            face_frames.extend([face_frames[-1]] * (45 - len(face_frames)))
        body_frames = [os.path.join(body_frames, img) for img in os.listdir(body_frames) if img.endswith('.jpg')]
        if len(body_frames)!=45:
            body_frames.extend([body_frames[-1]] * (45 - len(body_frames)))


        frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        front_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        left_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        right_frames.sort(key=lambda x: int(os.path.basename(x).split('.')[0]))
        face_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))
        body_frames.sort(key=lambda x: int(os.path.basename(x).split('_')[0]))




        buffer, buffer_front, buffer_left, buffer_right, keypoints_list, buffer_face, buffer_body= [], [], [], [], [], [], []
        posture_list, gesture_list = [], [] 

        for i, frame_name in enumerate(frames):
            if not i == 0 and not i % 3 == 2:
                continue
            if i >= 45:
                break

            img = cv2.imread(frame_name)
            front_img = cv2.imread(front_frames[i])
            left_img = cv2.imread(left_frames[i])
            right_img = cv2.imread(right_frames[i])

            img_face = cv2.imread(face_frames[i])
            img_body = cv2.imread(body_frames[i])
            keypoints = np.array(pose_list[i]['result'][0]['keypoints']).reshape(-1, 3)
            # keypoints_list.append(torch.from_numpy(keypoints).float())
            # keypoint=keypoint[94:115]
            # posture =  keypoint[:,:,:,:26,:]
            # gesture = keypoint[:,:,:,94:,:]            
            posture =  keypoints[:26]
            gesture = keypoints[94:136]
            
            posture_list.append(posture)
            gesture_list.append(gesture)
            
            
            
            # img_body = img[int(body[1]):int(body[1] + max(body[3], 20)), int(body[0]):int(body[0] + max(body[2], 10))]
            # img_face = img[int(face[1]):int(face[1] + max(face[3], 10)), int(face[0]):int(face[0] + max(face[2], 10))]

            if img.shape[0] != self.resize_height or img.shape[1] != self.resize_width:
                img = cv2.resize(img, (self.resize_width, self.resize_height))
            if front_img.shape[0] != self.resize_height or front_img.shape[1] != self.resize_width:
                front_img = cv2.resize(front_img, (self.resize_width, self.resize_height))
            if left_img.shape[0] != self.resize_height or left_img.shape[1] != self.resize_width:
                left_img = cv2.resize(left_img, (self.resize_width, self.resize_height))
            if right_img.shape[0] != self.resize_height or right_img.shape[1] != self.resize_width:
                right_img = cv2.resize(right_img, (self.resize_width, self.resize_height))

            if img_body.shape[0] != self.resize_height or img_body.shape[1] != self.resize_width:
                img_body = cv2.resize(img_body, (self.resize_width, self.resize_height))

            try:
                if img_face.shape[0] != self.face_height or img_face.shape[1] != self.face_width:
                    img_face = cv2.resize(img_face, (self.face_width, self.face_height))
                # if img_face.shape[0] != self.resize_height or img_face.shape[1] != self.resize_width:
                #     img_face = cv2.resize(img_face, (self.resize_width, self.resize_height))
            except:
                img_face = img_body

            buffer.append(torch.from_numpy(img).float())
            buffer_front.append(torch.from_numpy(front_img).float())
            buffer_left.append(torch.from_numpy(left_img).float())
            buffer_right.append(torch.from_numpy(right_img).float())

            buffer_body.append(torch.from_numpy(img_body).float())
            # if len(face_frames)==45:
            buffer_face.append(torch.from_numpy(img_face).float())
            # keypoints.append(torch.from_numpy(keypoints).float())
            # keypoints_tensor = torch.stack(keypoints_list)
            # posture_tensor = torch.tensor(posture_list, dtype=torch.float)
            # gesture_tensor = torch.tensor(gesture_list, dtype=torch.float)

            posture_array = np.array(posture_list, dtype=np.float32)  # 将 posture_list 轈换为 numpy 数组
            gesture_array = np.array(gesture_list, dtype=np.float32)  # 将 gesture_list 转换为 numpy 数组
            posture_tensor = torch.from_numpy(posture_array)  # 将 numpy 数组转换为 PyTorch 张量
            gesture_tensor = torch.from_numpy(gesture_array)  # 将 numpy 数组转换为 PyTorch 张量

        return torch.stack(buffer), torch.stack(buffer_front), torch.stack(buffer_left), torch.stack(
            buffer_right), torch.stack(buffer_face), torch.stack(buffer_body), posture_tensor,gesture_tensor
    #keypoints_tensor#torch.stack(keypoints) 

    # 随机翻转输入的PyTorch张量buffer(数据增强操作)
    def randomflip(self, buffer):

        # 以50%的概率在第二个维度上进行水平翻转
        if np.random.random() < 0.5:
            buffer = torch.flip(buffer, dims=[1])

        # 以50%的概率在第三个维度上进行垂直翻转
        if np.random.random() < 0.5:
            buffer = torch.flip(buffer, dims=[2])

        # 返回翻转后的张量buffer
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame

        return buffer

    def to_tensor(self, buffer):

        return buffer.permute(3, 0, 1, 2).contiguous()


import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(20)

train_dataset = CarDataset(csv_file='/root/training.csv')  # 'training.csv'
val_dataset = CarDataset(csv_file='/root/validation.csv')
test_dataset = CarDataset(csv_file='/root/testing.csv')

train_dataloader = DataLoader(train_dataset, batch_size=48, shuffle=True, num_workers=4, drop_last=False)
val_dataloader = DataLoader(val_dataset, batch_size=64,shuffle=False, num_workers=4, drop_last=False)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0, drop_last=False)


    
    
class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56,
                 stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        # self.bn_qk = nn.BatchNorm2d(groups)
        # self.bn_qr = nn.BatchNorm2d(groups)
        # self.bn_kr = nn.BatchNorm2d(groups)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):
        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)
        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                          width=True)
        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class AxialAttentionNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=True,
                 groups=8, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, s=1):
        super(AxialAttentionNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = int(64 * s)
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # 改：   # 添加一个新的卷积层来降低通道数
        # self.conv_reduce = nn.Conv2d(48, 3, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn_reduce = nn.BatchNorm2d(3)
        # self.conv1 = nn.Conv2d(48, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.conv1 = nn.Conv2d(48, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, int(128 * s), layers[0], kernel_size=56)
        self.layer2 = self._make_layer(block, int(256 * s), layers[1], stride=2, kernel_size=56,
                                       dilate=replace_stride_with_dilation[0])

        self.layer3 = self._make_layer(block, int(512 * s), layers[2], stride=2, kernel_size=28,
                                       dilate=replace_stride_with_dilation[1])

        self.avgpool = nn.AdaptiveAvgPool2d( (14, 14) )
        self.fc = nn.Linear(14,14)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                if isinstance(m, qkv_transform):
                    pass
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, AxialBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, kernel_size=56, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, groups=self.groups,
                            base_width=self.base_width, dilation=previous_dilation,
                            norm_layer=norm_layer, kernel_size=kernel_size))
        self.inplanes = planes * block.expansion
        if stride != 1:
            kernel_size = kernel_size // 2

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, kernel_size=kernel_size))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]

        # x = self.conv_reduce(x)
        # x = self.bn_reduce(x)
        # x = self.relu(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def PMANet_S(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [1, 2, 4, 1], s=0.5, **kwargs)
    return model


def PMANet_M(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.5, **kwargs)
    return model


def PMANet_L(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=0.75, **kwargs)
    return model


def PMANet_XL(pretrained=False, **kwargs):
    model = AxialAttentionNet(AxialBlock, [3, 4, 6, 3], s=1, **kwargs)
    return model
        
    
class TotalNet(nn.Module):
    def __init__(self):
        super(TotalNet, self).__init__()
        self.subnet1 = PMANet_S(pretrained=False)
        self.subnet2 = PMANet_S(pretrained=False)
        self.subnet3 = PMANet_S(pretrained=False)
        self.subnet4 = PMANet_S(pretrained=False)
        
        self.subnet5 = ImageConvNet_face() #ImageConvNet()
        self.subnet6 = ImageConvNet_body()# ResNet(Bottleneck,[3,4,6,3],4)

#         self.st_gcn1 = ST_GCN_18()
#         self.st_gcn2 = ST_GCN_18()
        self.conv3d_gesture = ConvNet3D(num_keypoints=26)
        self.conv3d_posture = ConvNet3D(num_keypoints=42)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        combined_features = 512
        
        self.aff_module_for_people_scenes = RDF(channels=512)
        self.aff_module_for_fused_points = RDF(channels=512)
        
        self.fc_split = nn.Linear(combined_features, 256)  # 4 * 64 = 256
        self.fc_fused = nn.Linear(combined_features, 256)  # 4 * 64 = 256

        
        self.fc1 = nn.Linear(256, 5) #512
        self.fc2 = nn.Linear(256, 7)
        self.fc3 = nn.Linear(256, 3)
        self.fc4 = nn.Linear(256, 5)

    def forward(self, img1,img2,img3,img4,face,body,gesture,posture):

        # print('image shape:', img1.shape)
        h1 = self.subnet1(img1)
        h2 = self.subnet2(img2)
        h3 = self.subnet3(img3)
        h4 = self.subnet4(img4)
        
        h_face = self.subnet5(face)
        h_body = self.subnet6(body)
        
        # h_gesture = self.st_gcn1(gesture)
        # h_posture = self.st_gcn2(posture)
        h_face = F.interpolate(h_face, size=(h1.size(2), h1.size(3)), mode='bilinear', align_corners=True)
        h_body = F.interpolate(h_body, size=(h1.size(2), h1.size(3)), mode='bilinear', align_corners=True)
#         h_face = h_face[:, :, :, 0]  
#         h_body = h_body[:, :, :, 0] 
        
#         # 选择最后一个维度的第一个切片
#         h1 = h1[:, :, :, 0]  # [32, 512, 14, 14] -> [32, 512, 14]
#         h2 = h2[:, :, :, 0]  # [32, 512, 14, 14] -> [32, 512, 14]
#         h3 = h3[:, :, :, 0]  # [32, 512, 14, 14] -> [32, 512, 14]
#         h4 = h4[:, :, :, 0]  # [32, 512, 14, 14] -> [32, 512, 14]


        h_gesture = self.conv3d_gesture(gesture)
        h_posture = self.conv3d_posture(posture)  
        h_gesture = h_gesture.unsqueeze(-1).unsqueeze(-1)  #
        h_posture = h_posture.unsqueeze(-1).unsqueeze(-1)  # 形状变为 (32, 512, 1, 1)
        h_gesture = h_gesture.expand(-1, -1, 14, 14)  # 扩展到14x14以匹配h1的大小
        h_posture = h_posture.expand(-1, -1, 14, 14)  # 扩展到14x14以匹配h1的大小
        # h_gesture = h_gesture.expand(-1, -1, 14)#, 14)  # 扩展到14x14以匹配h1的大小
        # h_posture = h_posture.expand(-1, -1, 14)#, 14)  # 扩展到14x14以匹配h1的大小
        
        # print('h1 shape:', h1.shape)
        # print('h2 shape:', h2.shape)
        # print('h3 shape:', h3.shape)
        # print('h4 shape:', h4.shape)
        # print('h_face shape:', h_face.shape)
        # print('h_body shape:', h_body.shape)
        # print('h_gesture shape:', h_gesture.shape)
        # print('h_posture shape:', h_posture.shape)
        
        # h1 = self.avg_pool(h1).view(h1.size(0), -1)
        # h2 = self.avg_pool(h2).view(h2.size(0), -1)
        # h3 = self.avg_pool(h3).view(h3.size(0), -1)
        # h4 = self.avg_pool(h4).view(h4.size(0), -1)
        # h_face = self.avg_pool(h_face).view(h_face.size(0), -1)
        # h_body = self.avg_pool(h_body).view(h_body.size(0), -1)
        # h_gesture = self.avg_pool(h_gesture).view(h_gesture.size(0), -1)
        # h_posture = self.avg_pool(h_posture).view(h_posture.size(0), -1)

    # 沿通道维度合并所有特征图
        # combined_features = torch.cat([h1, h2, h3, h4, h_face, h_body, h_gesture, h_posture], dim=1)
     
        # 通过加法合并特征
        x_people = h1 + h_face + h_body
        x_scenes = h2 + h3 + h4
        x_points = h_gesture + h_posture
        
        # print('x_people shape:', x_people.shape)
        # print('x_scenes shape:',x_scenes.shape)
        # print('x_points shape:', x_points.shape)      
        
       # 使用第一个AFF模块融合人和场景相关的特征
        x_fused_people_scenes = self.aff_module_for_people_scenes(x_people, x_scenes)
        # print('x_fused_people_scenes:',x_fused_people_scenes.shape)

        # 使用第二个AFF模块融合上一步的融合特征和关键点相关的特征
        x_fused_final = self.aff_module_for_fused_points(x_fused_people_scenes, x_points)
        
        # combined_features = self.fusion(h1, h2, h3, h4, h_face, h_body, h_gesture, h_posture)
        
        # print('x_fused_final shape:', x_fused_final.shape)
        
        x_fused_final = self.avg_pool(x_fused_final).view(x_fused_final.size(0), -1)
        
        
        # 使用新的全连接层将 x_fused_final 分成四个 64 维度的特征向量
        x_fused_split = self.fc_fused(x_fused_final).view(-1, 4, 64)
        x_fused_out1 = x_fused_split[:, 0, :]
        x_fused_out2 = x_fused_split[:, 1, :]
        x_fused_out3 = x_fused_split[:, 2, :]
        x_fused_out4 = x_fused_split[:, 3, :]

        # 拼接特征向量
        out1_features = torch.cat((x_fused_out1, x_fused_out2, x_fused_out3, x_fused_out4), dim=1)
        out2_features = torch.cat((x_fused_out2, x_fused_out1, x_fused_out3, x_fused_out4), dim=1)
        out3_features = torch.cat((x_fused_out3, x_fused_out1, x_fused_out2, x_fused_out4), dim=1)
        out4_features = torch.cat((x_fused_out4, x_fused_out1, x_fused_out2, x_fused_out3), dim=1)

        # 通过原有的全连接层计算输出
        out1 = self.fc1(out1_features)
        out2 = self.fc2(out2_features)
        out3 = self.fc3(out3_features)
        out4 = self.fc4(out4_features)

        return out1, out2, out3, out4
        


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

        
class ImageConvNet_body(nn.Module):

    def __init__(self):
        super(ImageConvNet_body, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.cnn1 = nn.Conv2d(192, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)

        # attention
        self.SeBlock1 = GLIBlock(64, 16, gamma=2, b=1)
        self.SeBlock2 = GLIBlock(64, 16, gamma=2, b=1)
        self.SeBlock3 = GLIBlock(128, 16, gamma=2, b=1)
        self.SeBlock4 = GLIBlock(128, 16, gamma=2, b=1)
        self.SeBlock5 = GLIBlock(256, 16, gamma=2, b=1)
        self.SeBlock6 = GLIBlock(256, 16, gamma=2, b=1)
        self.SeBlock7 = GLIBlock(512, 16, gamma=2, b=1)
        self.SeBlock8 = GLIBlock(512, 16, gamma=2, b=1)



    def forward(self, inp):
        c = F.relu(self.SeBlock1(self.bat10(self.cnn1(inp))))
        c = F.relu(self.SeBlock2(self.bat11(self.cnn2(c))))
        # c = self.pool(c)

        c = F.relu(self.SeBlock3(self.bat20(self.cnn3(c))))
        c = F.relu(self.SeBlock4(self.bat21(self.cnn4(c))))
        # c = self.pool(c)

        c = F.relu(self.SeBlock5(self.bat30(self.cnn5(c))))
        c = F.relu(self.SeBlock6(self.bat31(self.cnn6(c))))
        c = self.pool(c)

        c = F.relu(self.SeBlock7(self.bat40(self.cnn7(c))))
        c = F.relu(self.SeBlock8(self.bat41(self.cnn8(c))))

        return c


class ImageConvNet_face(nn.Module):

    def __init__(self):
        super(ImageConvNet_face, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.cnn1 = nn.Conv2d(48, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)

        # attention
        self.SeBlock1 = GLIBlock(64, 16, gamma=2, b=1)
        self.SeBlock2 = GLIBlock(64, 16, gamma=2, b=1)
        self.SeBlock3 = GLIBlock(128, 16, gamma=2, b=1)
        self.SeBlock4 = GLIBlock(128, 16, gamma=2, b=1)
        self.SeBlock5 = GLIBlock(256, 16, gamma=2, b=1)
        self.SeBlock6 = GLIBlock(256, 16, gamma=2, b=1)
        self.SeBlock7 = GLIBlock(512, 16, gamma=2, b=1)
        self.SeBlock8 = GLIBlock(512, 16, gamma=2, b=1)


    def forward(self, inp):
        c = F.relu(self.SeBlock1(self.bat10(self.cnn1(inp))))
        c = F.relu(self.SeBlock2(self.bat11(self.cnn2(c))))
        # c = self.pool(c)

        c = F.relu(self.SeBlock3(self.bat20(self.cnn3(c))))
        c = F.relu(self.SeBlock4(self.bat21(self.cnn4(c))))
        c = self.pool(c)

        c = F.relu(self.SeBlock5(self.bat30(self.cnn5(c))))
        c = F.relu(self.SeBlock6(self.bat31(self.cnn6(c))))
        c = self.pool(c)

        c = F.relu(self.SeBlock7(self.bat40(self.cnn7(c))))
        c = F.relu(self.SeBlock8(self.bat41(self.cnn8(c))))

        return c


class ConvNet3D(nn.Module):
    def __init__(self, num_classes=512, num_keypoints=42):
        super(ConvNet3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 1, 1))
        self.fc = nn.Linear(64 * 16 * (num_keypoints) * 1, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    
    
class ImageConvNet_Swin_block(nn.Module):

    def __init__(self):
        super(ImageConvNet_Swin_block, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.cnn1 = nn.Conv2d(192, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)
        # Swin Transformer Block
        # Swin Transformer 3D Block
        self.swin_block = SwinTransformerBlock3D(
            dim=512,  # 特征通道数，需要根据您的网络调整
            # depth=2,  # 这个阶段的深度
            num_heads=8,  # 注意力头的数量
            window_size=(1, 7, 7),  # 局部窗口大小
            mlp_ratio=4.0,  # MLP隐藏层维度与嵌入维度的比率
            qkv_bias=True,  # 是否添加可学习偏置到query, key, value
            qk_scale=None,  # 设置头维度的比例因子
            drop=0.0,  # Dropout比率
            attn_drop=0.0,  # 注意力权重的dropout比率
            drop_path=0.0,  # 随机深度比率
            norm_layer=nn.LayerNorm,  # 归一化层类型
            # downsample=None  # 层末尾的下采样模块，如果需要
            
        )
        # attention
        self.SeBlock1 = GLIBlock(64, 16, gamma=2, b=1)
        self.SeBlock2 = GLIBlock(64, 16, gamma=2, b=1)
        self.SeBlock3 = GLIBlock(128, 16, gamma=2, b=1)
        self.SeBlock4 = GLIBlock(128, 16, gamma=2, b=1)
        self.SeBlock5 = GLIBlock(256, 16, gamma=2, b=1)
        self.SeBlock6 = GLIBlock(256, 16, gamma=2, b=1)
        self.SeBlock7 = GLIBlock(512, 16, gamma=2, b=1)
        self.SeBlock8 = GLIBlock(512, 16, gamma=2, b=1)

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #
        # self.fc1 = nn.Linear(512, 5)
        # self.fc2 = nn.Linear(512, 7)
        # self.fc3 = nn.Linear(512, 3)
        # self.fc4 = nn.Linear(512, 5)

    def forward(self, inp):
        c = F.relu(self.SeBlock1(self.bat10(self.cnn1(inp))))
        c = F.relu(self.SeBlock2(self.bat11(self.cnn2(c))))
        c = self.pool(c)

        c = F.relu(self.SeBlock3(self.bat20(self.cnn3(c))))
        c = F.relu(self.SeBlock4(self.bat21(self.cnn4(c))))
        c = self.pool(c)

        c = F.relu(self.SeBlock5(self.bat30(self.cnn5(c))))
        c = F.relu(self.SeBlock6(self.bat31(self.cnn6(c))))
        c = self.pool(c)

        c = F.relu(self.SeBlock7(self.bat40(self.cnn7(c))))
        c = F.relu(self.SeBlock8(self.bat41(self.cnn8(c))))
        swin_output = self.swin_block(c.unsqueeze(0))  # 可能需要调整输入的形状

        # 结合 Swin Transformer 的输出
        # 可以选择如何融合 Swin Transformer 的输出
        c = c + swin_output.squeeze(0)  # 根据需要调整形状
        return c
    
    
class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=1
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv2d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm2d(planes)

        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)

        self.conv3=nn.Conv2d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        #判断残差有没有卷积
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        #参差数据
        residual=x

        #卷积操作
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)
        out=self.relu(out)

        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual=self.downsample(x)

        #将残差部分和卷积部分相加
        out=out+residual
        out=self.relu(out)

        return out




choices = ["demo", "main", "test", "checkValidation", "getVideoEmbeddings", "generateEmbeddingsForVideoAudio",
           "imageToImageQueries", "crossModalQueries"]

# parser = argparse.ArgumentParser(description="Select code to run.")
# parser.add_argument('--mode', default="test", choices=choices, type=str)

checkpoint_dir = '/root/AIDE'


class valConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        f1_list = []
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            # Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            f1_list.append(F1)
        return f1_list


class testConfusionMatrix(object):
    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))
        self.num_classes = num_classes
        self.labels = labels

    def update(self, preds, labels):
        for p, t in zip(preds, labels):
            self.matrix[p, t] += 1

    def summary(self):
        # calculate accuracy
        sum_TP = 0
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]
        acc = sum_TP / np.sum(self.matrix)
        print("the model accuracy is ", acc)

        # precision, recall, specificity
        table = PrettyTable()
        table.field_names = ["", "Precision", "Recall", "Specificity", "F1"]
        for i in range(self.num_classes):
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN
            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.
            F1 = round(2 * Precision * Recall / (Precision + Recall), 3) if Precision + Recall != 0 else 0.
            table.add_row([self.labels[i], Precision, Recall, Specificity, F1])
        print(table)

    def plot(self):
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()


class LossAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class AccAverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def getacc(self):
        return (self.sum * 100) / self.count


# Main function here
def main(use_cuda=True, EPOCHS=125, batch_size=48):
    # model = ImageConvNet().cuda()
    # model = TotalNet().cuda()
    model = TotalNet()  # 创建模型实例
    model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
    
    # # model.load_state_dict({k.replace('module.',''):v for k,v in model_dict.items()})
    # model.load_state_dict(model_dict, strict=False)
    
    model = model.cuda()  # 将模型移动到 CUDA 上



    crossEntropy1 = nn.CrossEntropyLoss()
    crossEntropy2 = nn.CrossEntropyLoss()
    crossEntropy3 = nn.CrossEntropyLoss()
    crossEntropy4 = nn.CrossEntropyLoss()
    print("Loaded dataloader and loss function.")

    # optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
    # optim = SGD(model.parameters(), lr=0.25e-3, momentum=0.9, weight_decay=1e-4)
    optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
    print("Optimizer loaded.")
    model.train()

    # state_dict = torch.load(os.path.join(checkpoint_dir, model_name))
    # model.load_state_dict(state_dict)
    # try:
    #     best_precision = 0
    #     lowest_loss = 100000
    #     best_avgf1 = 0
    #     # best_weightf1 = 0
    #     for epoch in range(EPOCHS):
    #         if (50 <= epoch < 100):
    #             optim = SGD(model.parameters(), lr=0.25e-4, momentum=0.9, weight_decay=1e-4)
    #         if (epoch >= 100):
    #             optim = SGD(model.parameters(), lr=0.25e-5, momentum=0.9, weight_decay=1e-4)
    # try:
    best_precision = 0
    lowest_loss = 100000
    best_avgf1 = 0
    best_weightf1 = 0


    for epoch in range(EPOCHS):
        if ( epoch <= 25):
            optim = SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        if (25 < epoch <= 50):
            optim = SGD(model.parameters(), lr=0.5e-3, momentum=0.9, weight_decay=1e-4)
        if (epoch > 50):
            optim = SGD(model.parameters(), lr=0.5e-4, momentum=0.9, weight_decay=1e-4)
        # Run algo

        # 训练损失
        train_losses = LossAverageMeter()

        # train_acc1 是一个AccAverageMeter类的实例，用于追踪训练准确率的平均值
        train_acc1 = AccAverageMeter()
        train_acc2 = AccAverageMeter()
        train_acc3 = AccAverageMeter()
        train_acc4 = AccAverageMeter()
        if (epoch == 0):
            end = time.time()
        # for subepoch, (img, aud, out) in enumerate(train_dataloader):#dataloader context,behavior_label

        # 遍历训练数据加载器，获取图像img1, img2, img3, img4和相应标签 , points
        for subepoch, (img1,img2,img3,img4,face,body,gesture,posture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(
                train_dataloader):

            # 打印第一个epoch中第一个批次所用时间
            if (epoch == 0 and subepoch == 0):
                print(time.time() - end)
            # 梯度清零
            optim.zero_grad()
            # 改变图像张量形状（五维->四维），使其与模型兼容
            B, _, _, H, W = img1.shape
            img1 = img1.view(B, -1,  H, W)  # [16, 3, 16, 224, 224]
            img2 = img2.view(B, -1,  H, W)
            img3 = img3.view(B, -1,  H, W)
            img4 = img4.view(B, -1,  H, W)
            
            face = face.view(B, -1,  64, 64)
            body = body.view(B, -1,  112,112)
# Gesture Skeleton Keypoint 3 (C)×16 (F)×42 (K)×1 (P)
# Posture Skeleton Keypoint 3 (C)×16 (F)×26 (K)×1 (P)
            gesture = gesture.view(B, 3,16,  26, 1)
            posture = posture.view(B,3,16, 42, 1)    
            # 批次大小
            M = img1.shape[0]
            # 将数据移到GPU上
            if use_cuda:
                img1 = img1.cuda()
                img2 = img2.cuda()
                img3 = img3.cuda()
                img4 = img4.cuda()
                
                face = face.cuda()
                body = body.cuda()
                
                gesture = gesture.cuda()
                posture = posture.cuda()
                
            emotion_label = emotion_label.cuda()
            behavior_label = behavior_label.cuda()
            context_label = context_label.cuda()
            vehicle_label = vehicle_label.cuda()

            # 将输入图像传递到模型(前向传播)
            out1, out2, out3, out4 = model(img1,img2,img3,img4,face,body,gesture,posture)
            # if subepoch%400 == 0:
            # 	print(o)
            # 	print(out)
            # print(o.shape, out.shape)
            # print(out1.shape, emotion_label.shape)

            # 计算单独损失和总损失
            loss1 = crossEntropy1(out1, emotion_label)
            # print(out2.shape, behavior_label.shape)
            loss2 = crossEntropy2(out2, behavior_label)
            # print(out3.shape, context_label.shape)
            loss3 = crossEntropy3(out3, context_label)
            # print(out4.shape, vehicle_label.shape)
            loss4 = crossEntropy4(out4, vehicle_label)
            loss = loss1 + loss2 + loss3 + loss4
            # print(loss)

            # 更新训练损失
            train_losses.update(loss.item(), M)
            # 反向传播，进一步优化
            loss.backward()
            optim.step()

            # Calculate accuracy
            out1 = F.softmax(out1, 1)  # Softmax将一组数值转换为概率分布
            ind = out1.argmax(dim=1)  # ind保存每行中最大值所在的索引，即概率最大的类别
            # print(ind.data)
            # print(out1.data)

            # 计算当前批次的准确率
            accuracy1 = (ind.data == emotion_label.data).sum() * 1.0 / M
            # 更新整个训练过程的准确率平均值
            train_acc1.update((ind.data == emotion_label.data).sum() * 1.0, M)

            out2 = F.softmax(out2, 1)
            ind = out2.argmax(dim=1)
            accuracy2 = (ind.data == behavior_label.data).sum() * 1.0 / M
            train_acc2.update((ind.data == behavior_label.data).sum() * 1.0, M)

            out3 = F.softmax(out3, 1)
            ind = out3.argmax(dim=1)
            accuracy3 = (ind.data == context_label.data).sum() * 1.0 / M
            train_acc3.update((ind.data == context_label.data).sum() * 1.0, M)

            out4 = F.softmax(out4, 1)
            ind = out4.argmax(dim=1)
            accuracy4 = (ind.data == vehicle_label.data).sum() * 1.0 / M
            train_acc4.update((ind.data == vehicle_label.data).sum() * 1.0, M)


            # if subepoch % 1 == 0:
            print("Epoch: %d, Subepoch: %d, Loss: %f, "
                    "batch_size: %d, total_acc1: %f, total_acc2: %f, total_acc3: %f, total_acc4: %f" % (
                epoch, subepoch, train_losses.avg, M,
                train_acc1.getacc(),
                train_acc2.getacc(),
                train_acc3.getacc(),
                train_acc4.getacc()))

            with open(file="/root/AIDE/CNNTrans_aff2.txt", mode="a+") as f:
                f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, total_acc1: %f,total_acc2: %f, total_acc3: %f, total_acc4: %f"\
                     %(epoch, subepoch, train_losses.avg, M, train_acc1.getacc(), train_acc2.getacc(), train_acc3.getacc(), train_acc4.getacc()))


        # 验证阶段
        print("Valing...")
        # val_losses = LossAverageMeter()

        val_losses1 = LossAverageMeter()
        val_losses2 = LossAverageMeter()
        val_losses3 = LossAverageMeter()
        val_losses4 = LossAverageMeter()
        # val_losses = (val_losses1.avg + val_losses2.avg + val_losses3.avg + val_losses4.avg) / 4.0

        val_acc1 = AccAverageMeter()
        val_acc2 = AccAverageMeter()
        val_acc3 = AccAverageMeter()
        val_acc4 = AccAverageMeter()
  
        # 混淆矩阵
        valconfusion1 = valConfusionMatrix(num_classes = 5, labels = EMOTION_LABEL)
        valconfusion2 = valConfusionMatrix(num_classes = 7, labels = DRIVER_BEHAVIOR_LABEL)
        valconfusion3 = valConfusionMatrix(num_classes = 3, labels = SCENE_CENTRIC_CONTEXT_LABEL)
        valconfusion4 = valConfusionMatrix(num_classes = 5, labels = VEHICLE_BASED_CONTEXT_LABEL)

        # 将模型设置为评估模式
        model.eval()


        for subepoch1, (img1,img2,img3,img4,face,body,gesture,posture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(
                val_dataloader):

            if (epoch == 0 and subepoch1 == 0):
                print(time.time() - end)
            with torch.no_grad():
                             
                B, _, _, H, W = img1.shape
                img1 = img1.view(B, -1,  H, W)  # [16, 3, 16, 224, 224]
                img2 = img2.view(B, -1,  H, W)
                img3 = img3.view(B, -1,  H, W)
                img4 = img4.view(B, -1,  H, W)

                face = face.view(B, -1,  64, 64)
                body = body.view(B, -1,  112,112)
# Gesture Skeleton Keypoint 3 (C)×16 (F)×42 (K)×1 (P)
# Posture Skeleton Keypoint 3 (C)×16 (F)×26 (K)×1 (P)
                gesture = gesture.view(B, 3,16,  26, 1)
                posture = posture.view(B,3,16, 42, 1)    
                # gesture = gesture.view(B, -1,  26, 1)
                # posture = posture.view(B,-1, 21, 1)                 

                # 批次大小
                M = img1.shape[0]
                # 将数据移到GPU上
                if use_cuda:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    img3 = img3.cuda()
                    img4 = img4.cuda()

                    face = face.cuda()
                    body = body.cuda()

                    gesture = gesture.cuda()
                    posture = posture.cuda()

                emotion_label = emotion_label.cuda()
                behavior_label = behavior_label.cuda()
                context_label = context_label.cuda()
                vehicle_label = vehicle_label.cuda()

                # 将输入图像传递到模型(前向传播)
                out1, out2, out3, out4 = model(img1,img2,img3,img4,face,body,gesture,posture)
               

                loss1 = crossEntropy1(out1, emotion_label)
                # print(out2.shape, behavior_label.shape)
                loss2 = crossEntropy2(out2, behavior_label)
                # print(out3.shape, context_label.shape)
                loss3 = crossEntropy3(out3, context_label)
                # print(out4.shape, vehicle_label.shape)
                loss4 = crossEntropy4(out4, vehicle_label)
                loss = loss1 + loss2 + loss3 + loss4
                # print(loss)
                val_losses1.update(loss1.item(), M)
                val_losses2.update(loss1.item(), M)
                val_losses3.update(loss1.item(), M)
                val_losses4.update(loss1.item(), M)

                val_losses = (val_losses1.avg + val_losses2.avg + val_losses3.avg + val_losses4.avg) / 4.0
                #早停，防止过拟合
                early_stopping = EarlyStopping(7, verbose=True)
                early_stopping(val_losses, model)
                # 若满足 early stopping 要求
                if early_stopping.early_stop:
                    print("Early stopping")
                    # 结束模型训练
                    break
                # Calculate accuracy
                out1 = F.softmax(out1, 1)
                ind1 = out1.argmax(dim=1)
                # print(ind.data)
                # print(out1.data)
                accuracy1 = (ind1.data == emotion_label.data).sum() * 1.0 / M
                val_acc1.update((ind1.data == emotion_label.data).sum() * 1.0, M)
                valconfusion1.update(ind1.to("cpu").numpy(), emotion_label.to("cpu").numpy())  # 更新混淆矩阵
                avgf11 = (valconfusion1.summary()[0] + valconfusion1.summary()[1] + valconfusion1.summary()[2]+
                         valconfusion1.summary()[3]+valconfusion1.summary()[4]) / 5.0

                out2 = F.softmax(out2, 1)
                ind2 = out2.argmax(dim=1)
                accuracy2 = (ind2.data == behavior_label.data).sum() * 1.0 / M
                val_acc2.update((ind2.data == behavior_label.data).sum() * 1.0, M)
                valconfusion2.update(ind2.to("cpu").numpy(), behavior_label.to("cpu").numpy())
                avgf12 = (valconfusion2.summary()[0] + valconfusion2.summary()[1] + valconfusion2.summary()[2] +
                         valconfusion2.summary()[3] + valconfusion2.summary()[4] +
                         valconfusion2.summary()[5] + valconfusion2.summary()[6]) / 7.0

                out3 = F.softmax(out3, 1)
                ind3 = out3.argmax(dim=1)
                accuracy3 = (ind3.data == context_label.data).sum() * 1.0 / M
                val_acc3.update((ind3.data == context_label.data).sum() * 1.0, M)
                valconfusion3.update(ind3.to("cpu").numpy(), context_label.to("cpu").numpy())
                avgf13 = (valconfusion3.summary()[0] + valconfusion3.summary()[1] + valconfusion3.summary()[2]) / 3.0

                out4 = F.softmax(out4, 1)
                ind4 = out4.argmax(dim=1)
                accuracy4 = (ind4.data == vehicle_label.data).sum() * 1.0 / M
                val_acc4.update((ind4.data == vehicle_label.data).sum() * 1.0, M)
                valconfusion4.update(ind4.to("cpu").numpy(), vehicle_label.to("cpu").numpy())
                avgf14 = (valconfusion4.summary()[0] + valconfusion4.summary()[1] + valconfusion4.summary()[2] +
                         valconfusion4.summary()[3] + valconfusion4.summary()[4]) / 5.0

                total_avgf1 = (avgf11 + avgf12 + avgf13 + avgf14) / 4.0

                # if subepoch1 % 1 == 0:
                print(
                    "Val Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d,total_acc1: %f, total_acc2: %f, "
                    "total_acc3: %f, total_acc4: %f, avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f" % (
                        epoch, subepoch1, val_losses, M,
                        val_acc1.getacc(),
                        val_acc2.getacc(),
                        val_acc3.getacc(),
                        val_acc4.getacc(), avgf11, avgf12, avgf13, avgf14))
                
                with open(file="/root/AIDE/val_CNNTrans_aff2.txt", mode="a+") as f:
                    f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, total_acc1: %f,total_acc2: %f, total_acc3: %f, total_acc4: %f, \
                            avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f\n"\
                     %(epoch, subepoch1, val_losses, M, val_acc1.getacc(), val_acc2.getacc(), val_acc3.getacc(), val_acc4.getacc(),avgf11, avgf12, avgf13, avgf14))

        # 更新最佳模型
        is_best_avgf1 = total_avgf1 > best_avgf1
        # is_best_weightf1 = weightf1 > best_weightf1
        val_acc = (val_acc1.getacc() + val_acc2.getacc() + val_acc3.getacc() + val_acc4.getacc()) / 4.0
        is_best = val_acc > best_precision
        is_lowest_loss = val_losses < lowest_loss
        best_precision = max(val_acc, best_precision)
        lowest_loss = min(val_losses, lowest_loss)
        best_avgf1 = max(total_avgf1, best_avgf1)


        print("Epoch: %d,best_precision: %f,lowest_loss: %f,best_avgf1: %f" % (
        epoch, best_precision, lowest_loss, best_avgf1))


        # 保存最佳模型(将当前模型的权重保存到best_model.pt文件中)
        best_path = os.path.join(checkpoint_dir, 'best_model_CNNTrans_aff_out2.pt')
        if is_best:
            with open(file="/root/val_CNNTrans_aff2.txt", mode="w") as f:
                    f.write("Epoch: %d, Subepoch: %d, Loss: %f, batch_size: %d, total_acc1: %f,total_acc2: %f, total_acc3: %f, total_acc4: %f, \
                            avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f\n"\
                     %(epoch, subepoch1, val_losses, M, val_acc1.getacc(), val_acc2.getacc(), val_acc3.getacc(), val_acc4.getacc(),avgf11, avgf12, avgf13, avgf14))
            # shutil.copyfile(save_path, best_path)
            torch.save(model.state_dict(), best_path)
            print("Successfully saved the model with the best precision!")

        # 保存最低损失模型
        # lowest_path = os.path.join(checkpoint_dir, 'lowest_loss_swin_block_context.pt')
        # if is_lowest_loss:
        #     shutil.copyfile(save_path, lowest_path)
            # torch.save(model.state_dict(), lowest_path)
            print("Successfully saved the model with the lowest loss!")

        # 保存最佳平均F1模型
        # best_avgf1_path = os.path.join(checkpoint_dir, 'best_avgf1_swin_block_context.pt')
        # if is_best_avgf1:
        #     shutil.copyfile(save_path, best_avgf1_path)
            # torch.save(model.state_dict(), best_avgf1_path)
            print("Successfully saved the model with the best avgf1!")



class TestMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n

    def getacc(self):
        return (self.sum * 100) / self.count


def test(use_cuda=True, batch_size=16, model_name="/root/AIDE/best_model_CNNTrans_aff_out2.pt"):

    model = TotalNet()  # 创建模型实例
    model = nn.DataParallel(model)  # 使用 DataParallel 包装模型
    model = model.cuda()  # 将模型移动到 CUDA 上
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name))
        print("Loading from previous checkpoint.")
        

    test_dataset = CarDataset(csv_file='/root/testing.csv')

    testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    crossEntropy = nn.CrossEntropyLoss()
    print("Loaded dataloader and loss function.")

    test_losses = LossAverageMeter()
    test_acc1 = TestMeter()
    test_acc2 = TestMeter()
    test_acc3 = TestMeter()
    test_acc4 = TestMeter()


    testconfusion1 = valConfusionMatrix(num_classes=5, labels=EMOTION_LABEL)
    testconfusion2 = valConfusionMatrix(num_classes=7, labels=DRIVER_BEHAVIOR_LABEL)
    testconfusion3 = valConfusionMatrix(num_classes=3, labels=SCENE_CENTRIC_CONTEXT_LABEL)
    testconfusion4 = valConfusionMatrix(num_classes=5, labels=VEHICLE_BASED_CONTEXT_LABEL)

    model.eval()

    best_precision = 0
    lowest_loss = 100000
    best_avgf1 = 0
    for subepoch2, (img1,img2,img3,img4,face,body,gesture,posture,emotion_label, behavior_label, context_label, vehicle_label) in enumerate(
                test_dataloader):

        # if (epoch == 0 and subepoch2 == 0):
        #     print(time.time() - end)
        with torch.no_grad():

            B, _, _, H, W = img1.shape
            img1 = img1.view(B, -1,  H, W)  # [16, 3, 16, 224, 224]
            img2 = img2.view(B, -1,  H, W)
            img3 = img3.view(B, -1,  H, W)
            img4 = img4.view(B, -1,  H, W)

            face = face.view(B, -1,  64, 64)
            body = body.view(B, -1,  112,112)
# Gesture Skeleton Keypoint 3 (C)×16 (F)×42 (K)×1 (P)
# Posture Skeleton Keypoint 3 (C)×16 (F)×26 (K)×1 (P)
            gesture = gesture.view(B, 3,16,  26, 1)
            posture = posture.view(B,3,16, 42, 1)    
            # gesture = gesture.view(B, -1,  39, 1)
            # posture = posture.view(B,-1, 24, 1)             

            # 批次大小
            M = img1.shape[0]
            # 将数据移到GPU上
            if use_cuda:
                img1 = img1.cuda()
                img2 = img2.cuda()
                img3 = img3.cuda()
                img4 = img4.cuda()

                face = face.cuda()
                body = body.cuda()

                gesture = gesture.cuda()
                posture = posture.cuda()

            emotion_label = emotion_label.cuda()
            behavior_label = behavior_label.cuda()
            context_label = context_label.cuda()
            vehicle_label = vehicle_label.cuda()

            # 将输入图像传递到模型(前向传播)
            out1, out2, out3, out4 = model(img1,img2,img3,img4,face,body,gesture,posture)


            # Calculate accuracy
            out1 = F.softmax(out1, 1)
            ind1 = out1.argmax(dim=1)
            # print(ind.data)
            # print(out1.data)
            accuracy1 = (ind1.data == emotion_label.data).sum() * 1.0 / M
            test_acc1.update((ind1.data == emotion_label.data).sum() * 1.0, M)
            testconfusion1.update(ind1.to("cpu").numpy(), emotion_label.to("cpu").numpy())
            avgf11 = (testconfusion1.summary()[0] + testconfusion1.summary()[1] + testconfusion1.summary()[2] +
                      testconfusion1.summary()[3] + testconfusion1.summary()[4]) / 5.0

            out2 = F.softmax(out2, 1)
            ind2 = out2.argmax(dim=1)
            accuracy2 = (ind2.data == behavior_label.data).sum() * 1.0 / M
            test_acc2.update((ind2.data == behavior_label.data).sum() * 1.0, M)
            testconfusion2.update(ind2.to("cpu").numpy(), behavior_label.to("cpu").numpy())
            avgf12 = (testconfusion2.summary()[0] + testconfusion2.summary()[1] + testconfusion2.summary()[2] +
                      testconfusion2.summary()[3] + testconfusion2.summary()[4] +
                      testconfusion2.summary()[5] + testconfusion2.summary()[6]) / 7.0

            out3 = F.softmax(out3, 1)
            ind3 = out3.argmax(dim=1)
            accuracy3 = (ind3.data == context_label.data).sum() * 1.0 / M
            test_acc3.update((ind3.data == context_label.data).sum() * 1.0, M)
            testconfusion3.update(ind3.to("cpu").numpy(), context_label.to("cpu").numpy())
            avgf13 = (testconfusion3.summary()[0] + testconfusion3.summary()[1] + testconfusion3.summary()[2]) / 3.0

            out4 = F.softmax(out4, 1)
            ind4 = out4.argmax(dim=1)
            accuracy4 = (ind4.data == vehicle_label.data).sum() * 1.0 / M
            test_acc4.update((ind4.data == vehicle_label.data).sum() * 1.0, M)
            testconfusion4.update(ind4.to("cpu").numpy(), vehicle_label.to("cpu").numpy())
            avgf14 = (testconfusion4.summary()[0] + testconfusion4.summary()[1] + testconfusion4.summary()[2] +
                      testconfusion4.summary()[3] + testconfusion4.summary()[4]) / 5.0

            total_avgf1 = (avgf11 + avgf12 + avgf13 + avgf14) / 4.0

            # if subepoch1 % 1 == 0:
            print(
                "Test  Subepoch: %d, batch_size: %d,total_acc1: %f, total_acc2: %f, "
                "total_acc3: %f, total_acc4: %f, avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f" % (
                    subepoch2, M,
                    test_acc1.getacc(),
                    test_acc2.getacc(),
                    test_acc3.getacc(),
                    test_acc4.getacc(), avgf11, avgf12, avgf13, avgf14))
            
            with open(file="/root/AIDE/test_CNNTrans_aff_axialnet2.txt", mode="a+") as f:
                    f.write("Test  Subepoch: %d, batch_size: %d,total_acc1: %f, total_acc2: %f, "
                "total_acc3: %f, total_acc4: %f, avgf11: %f, avgf12: %f, avgf13: %f, avgf14: %f\n"\
                     %(subepoch2, M,
                    test_acc1.getacc(),
                    test_acc2.getacc(),
                    test_acc3.getacc(),
                    test_acc4.getacc(), avgf11, avgf12, avgf13, avgf14))


    testconfusion1.summary()
    testconfusion2.summary()
    testconfusion3.summary()
    testconfusion4.summary()
    # testconfusion1.plot()
    # testconfusion2.plot()
    # testconfusion3.plot()
    # testconfusion4.plot()



if __name__ == "__main__":
    cuda = True

mode = "train"
print("running...")

if mode == "train":
    main(use_cuda=cuda, batch_size=64)
    print("main mode...")
elif mode == "test":
    test(use_cuda=cuda, batch_size=64)
    print("test mode...")
