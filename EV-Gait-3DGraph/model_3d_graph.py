# -- coding: utf-8 --**
# GCN model

import torch

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import voxel_grid, max_pool, max_pool_x, GMMConv
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)



class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = GMMConv(in_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = GMMConv(out_channel, out_channel, dim=3, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)

        self.shortcut_conv = GMMConv(in_channel, out_channel, dim=3, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)

    def forward(self, data):
        data.x = F.elu(self.left_bn2(
            self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                            data.edge_index, data.edge_attr)) + self.shortcut_bn(
            self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))

        return data


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GMMConv(1, 64, dim=3, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)

        self.block1 = ResidualBlock(64, 128)
        self.block2 = ResidualBlock(128, 256)
        self.block3 = ResidualBlock(256, 512)

        self.fc1 = torch.nn.Linear(8 * 512, 1024)
        self.bn = torch.nn.BatchNorm1d(1024)
        # self.drop_out = torch.nn.Dropout()
        self.fc2 = torch.nn.Linear(1024, 20)

    def forward(self, data):
        data.x = F.elu(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        cluster = voxel_grid(data.pos, data.batch, size=4)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block1(data)
        cluster = voxel_grid(data.pos, data.batch, size=6)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block2(data)
        cluster = voxel_grid(data.pos, data.batch, size=24)
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))

        data = self.block3(data)
        cluster = voxel_grid(data.pos, data.batch, size=64)
        x = max_pool_x(cluster, data.x, data.batch, size=8)

        # if your torch-geometric version is below 1.3.2(roughly, we do not test all versions), use x.view() instead of x[0].view()
        # x = x.view(-1, self.fc1.weight.size(1))
        x = x[0].view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        # x = self.drop_out(x)
        x = self.bn(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)