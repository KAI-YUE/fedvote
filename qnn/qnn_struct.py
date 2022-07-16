import numpy as np

# PyTorch Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# My Libraries
from qnn import QuantizedNN
from qnn.qnn_blocks import *

class QuantizedMLP(QuantizedNN):
    def __init__(self, in_dims, out_dims, dim_hidden=200):
        super(QuantizedMLP, self).__init__()
        self.fc1 = QuantizedLinear(in_dims, dim_hidden)
        self.bn1 = nn.BatchNorm1d(dim_hidden, track_running_stats=False, affine=False)
        
        self.fc2 = QuantizedLinear(dim_hidden, dim_hidden)
        self.bn2 = nn.BatchNorm1d(dim_hidden, track_running_stats=False, affine=False)

        self.fc3 = nn.Linear(dim_hidden, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        
        out = F.relu(self.bn1(self.fc1(x)))
        out = F.relu(self.bn2(self.fc2(out)))
        out = self.fc3(out)
        return out

class QuantizedLeNet_BN(QuantizedNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(QuantizedLeNet_BN, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = QuantizedConv2d(self.in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = QuantizedConv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = QuantizedLinear(self.fc_input_size, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class QuantizedVGG_7(QuantizedNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(QuantizedVGG_7, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/8)**2 * 512
        
        self.conv1_1 = QuantizedConv2d(self.in_channels, 128, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.conv1_2 = QuantizedConv2d(128, 128, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = QuantizedConv2d(128, 256, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv2_2 = QuantizedConv2d(256, 256, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = QuantizedConv2d(256, 512, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv3_2 = QuantizedConv2d(512, 512, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.fc1 = QuantizedLinear(self.fc_input_size, 1024)
        self.bn4 = nn.BatchNorm1d(1024, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(1024, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.mp3(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn4(self.fc1(x)))
        # x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class QuantizedNeuralNet_BN(QuantizedNN):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(QuantizedNeuralNet_BN, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = QuantizedConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = QuantizedConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = QuantizedLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

class CompleteQuantizedNeuralNet(QuantizedNN):
    def __init__(self, in_dims, in_channels, out_dims=10):  
        super(CompleteQuantizedNeuralNet, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        
        self.conv1 = QuantizedConv2d(self.in_channels, 64, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = QuantizedConv2d(64, 128, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = QuantizedLinear(2048, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = QuantizedLinear(512, out_dims)

    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def freeze_final_layer(self):
        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

