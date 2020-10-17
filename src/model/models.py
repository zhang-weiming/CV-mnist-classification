import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



def convLayer(in_planes, out_planes, useDropout=False, activation=True):
    "3x3 convolution with padding"
    if activation:
        seq = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    else:
        seq = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    if useDropout:  # Add dropout module
        list_seq = list(seq.modules())[1:]
        list_seq.append(nn.Dropout(0.1))
        seq = nn.Sequential(*list_seq)
    return seq


class CosDistLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CosDistLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight, 0.0, np.sqrt(2.0/self.in_features))
        # nn.init.xavier_normal_(self.weight)
        nn.init.constant_(self.bias, 0)

    def cos_dist(self, matrix1: torch.Tensor, matrix2: torch.Tensor):
        matrix1_matrix2 = torch.mm(matrix1, matrix2.T)
        matrix1_norm = torch.sqrt(torch.mul(matrix1, matrix1).sum(1))
        matrix1_norm = matrix1_norm.unsqueeze(1)
        matrix2_norm = torch.sqrt(torch.mul(matrix2, matrix2).sum(1))
        matrix2_norm = matrix2_norm.unsqueeze(1)
        cosine_distance = torch.div(matrix1_matrix2, torch.mm(matrix1_norm, matrix2_norm.t()))
        return cosine_distance

    def forward(self, input):
        weight = F.normalize(self.weight.data, dim=1, p=2)
        input_norm = F.normalize(input, dim=1)
        # return self.cos_dist(input, self.weight)
        # ret = input_norm.matmul(weight.t())

        if input_norm.dim() == 2 and self.bias is not None:
            # fused op is marginally faster
            ret = torch.addmm(self.bias, input_norm, weight.t())
        else:
            output = input_norm.matmul(weight.t())
            if self.bias is not None:
                output += self.bias
            ret = output
        return ret

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


class Classifier(nn.Module):
    def __init__(self, layer_size, nClasses=0, num_channels=3, feat_dim=1024, useDropout=False, image_size=84):
        super(Classifier, self).__init__()

        """
        Builds a CNN to produce embeddings
        :param layer_sizes: A list of length 4 containing the layer sizes
        :param nClasses: If nClasses>0, we want a FC layer at the end with nClasses size.
        :param num_channels: Number of channels of images
        :param useDroput: use Dropout with p=0.1 in each Conv block
        """
        self.layer1 = convLayer(num_channels, layer_size, useDropout)
        self.layer2 = convLayer(layer_size, layer_size, useDropout)
        self.layer3 = convLayer(layer_size, layer_size, useDropout)
        self.layer4 = convLayer(layer_size, layer_size, useDropout)

        final_size = int(math.floor(image_size / (2 * 2 * 2 * 2)))
        self.out_size = final_size * final_size * layer_size
        # self.layer_fc = nn.Linear(self.out_size, feat_dim)

        if nClasses > 0:  # We want a linear
            self.useClassification = True
            self.layer5 = nn.Linear(self.out_size, nClasses)
            # self.layer5 = CosDistLayer(feat_dim, nClasses, bias=True)
            # self.out_size = nClasses
        else:
            self.useClassification = False

        self.weights_init(self.layer1)
        self.weights_init(self.layer2)
        self.weights_init(self.layer3)
        self.weights_init(self.layer4)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def forward(self, image_input):
        """
        Runs the CNN producing the embeddings and the gradients.
        :param image_input: Image input to produce embeddings for. [batch_size, 28, 28, 1]
        :return: Embeddings of size [batch_size, 64]
        """
        x = self.layer1(image_input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        # x = self.layer_fc(x)
        if self.useClassification:
            x = self.layer5(x)
        return x
