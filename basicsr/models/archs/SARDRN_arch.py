# encoding = utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class SARDRN(nn.Module):
    def __init__(self):
        super(SARDRN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv4_1 = nn.Conv2d(64, 64, kernel_size=3, padding=4, dilation=4)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, padding=3, dilation=3)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, padding=2, dilation=2)
        self.conv7 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        conv1_out = F.relu(self.conv1(x))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))

        # Short-cut connection
        conv4_out = conv3_out + conv1_out
        conv4_out = F.relu(self.conv4_1(conv4_out))

        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))

        # Another short-cut connection
        conv7_out = conv6_out + conv4_out
        conv7_out = self.conv7(conv7_out)

        return conv7_out
