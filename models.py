import torch
import torch.nn as nn


class NetWork(nn.Module):

    def __init__(self):
        super(NetWork, self).__init__()
        # print('lhsb')
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, stride=1)

    def forward(self, data):
        
        out = self.conv1(data)

        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        

        return out
