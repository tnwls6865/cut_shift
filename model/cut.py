'''ResNet18/34/50/101/152 in Pytorch.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.autograd import Variable


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def channel_mix(self, out, rand_index, ratio):

        channel = out.size(1)
        x = out.clone()
        channel = int(channel * ratio)
        temp = out[rand_index, channel:, :, :]
        x[:, channel:, :, :] = temp
        
        return out

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def channel_mix(self, out, rand_index, ratio):

        channel = out.size(1)
        channel = int(channel * ratio)
  
        temp = out[rand_index, channel:, :, :]
        out[:, channel:, :, :] = temp

        return out

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = F.relu(self.bn2(self.conv2(out)))
        #out = self.channel_mix(out, rand_)

        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        self.i = 0

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    '''def feature_mix (self, out, size1, size2, rand_index, bbox):
        out_gap = size2 /  size1
        bbx1, bby1, bbx2, bby2 =  int(bbox[0]*out_gap), int(bbox[1]*out_gap), int(bbox[2]*out_gap), int(bbox[3]*out_gap)
        out_a = out[rand_index, :, bbx1:bbx2, bby1:bby2]
        out[:, :, bbx1:bbx2, bby1:bby2] = out_a
        
        return out '''
    def channel_mix(self, out, rand_index, lam):

        out = out*lam + out[rand_index]*(1-lam)
        
        return out

    def forward(self, x, is_train, rand_index, lam=0):
        if is_train == True:
            layer_mix = random.randint(0,6)
            
            if layer_mix == 0:
                x = self.channel_mix(x, rand_index, lam)

            f = self.conv1(x)
            out = F.relu(self.bn1(self.conv1(x)))
            
            if layer_mix == 1:
                out = self.channel_mix(out, rand_index, lam)
            
            out = self.layer1(out) # b, 64, 32, 32
            
            if layer_mix == 2:
                out = self.channel_mix(out, rand_index, lam)
            
            out = self.layer2(out) # b, 128, 16, 16
            
            if layer_mix == 3:
                out = self.channel_mix(out, rand_index, lam)

            out = self.layer3(out) # b, 256, 8, 8
            
            if layer_mix == 4:
                out = self.channel_mix(out, rand_index, lam)
            
            out = self.layer4(out) # b, 512, 4, 4
            
            if layer_mix == 5:
                out = self.channel_mix(out, rand_index, lam)
    
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            if layer_mix == 6:
                out = self.channel_mix(out, rand_index, lam)

            out = self.linear(out)
            return out

        elif is_train == False:
            f = self.conv1(x)
            out = F.relu(self.bn1(self.conv1(x))) # b, 64, 32, 32
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)

            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.linear(out)
            return out


def ResNet18(num_classes=10):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes=10):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes=10):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes=10):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes=10):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)

def test_resnet():
    net = ResNet50()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test_resnet()