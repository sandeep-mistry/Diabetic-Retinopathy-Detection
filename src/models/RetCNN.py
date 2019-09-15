"""

This CNN returns predictions as specified from 'Convolutional Neural Networks for Diabetic Retinopathy' Frans Coenen et. al

"""
import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models

cn_kernel = (3, 3)
mp_kernel = (2, 2)
mp_stride = (2, 2)


class RetCNN(nn.Module):
    def __init__(self, feature=False):
        super(RetCNN, self).__init__()
        self.feature = feature
        self.conv1 = nn.Conv2d(3, 32, cn_kernel, padding=1)
        self.conv11 = nn.Conv2d(32, 32, cn_kernel, padding=1)
        self.conv2 = nn.Conv2d(32, 64, cn_kernel, padding=1)
        self.conv22 = nn.Conv2d(64, 64, cn_kernel, padding=1)
        self.conv3 = nn.Conv2d(64, 128, cn_kernel, padding=1)
        self.conv33 = nn.Conv2d(128, 128, cn_kernel, padding=1)
        self.conv4 = nn.Conv2d(128, 256, cn_kernel, padding=1)
        self.conv44 = nn.Conv2d(256, 256, cn_kernel, padding=1)
        self.conv5 = nn.Conv2d(256, 512, cn_kernel, padding=1)
        self.conv55 = nn.Conv2d(512, 512, cn_kernel, padding=(1, 0))

        self.fc1 = nn.Linear(1024, 1024)  # check for 512 input
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 5)

        # weight initialisations from a normal distribution mean:0 std:1
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.normal_(self.conv11.weight)
        torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.normal_(self.conv22.weight)
        torch.nn.init.normal_(self.conv3.weight)
        torch.nn.init.normal_(self.conv33.weight)
        torch.nn.init.normal_(self.conv4.weight)
        torch.nn.init.normal_(self.conv44.weight)
        torch.nn.init.normal_(self.conv5.weight)
        torch.nn.init.normal_(self.conv55.weight)
        torch.nn.init.normal_(self.fc1.weight)
        torch.nn.init.normal_(self.fc2.weight)
        torch.nn.init.normal_(self.fc3.weight)

        self.BN1 = nn.BatchNorm2d(32)
        self.BN2 = nn.BatchNorm2d(64)
        self.BN3 = nn.BatchNorm2d(128)
        self.BN4 = nn.BatchNorm2d(256)
        self.BN5 = nn.BatchNorm2d(512)

        self.mp2d = nn.MaxPool2d(mp_kernel, stride=mp_stride)
        self.dp = nn.Dropout(p=0.5)
        self.sm = nn.Softmax()

    def forward(self, x):
        out = self.conv1(x)
        out = F.leaky_relu(out)
        out = self.BN1(out)
        out = self.mp2d(out)

        out = self.conv11(out)
        out = F.leaky_relu(out)
        out = self.BN1(out)
        out = self.mp2d(out)

        out = self.conv2(out)
        out = F.leaky_relu(out)
        out = self.BN2(out)
        out = self.mp2d(out)

        out = self.conv22(out)
        out = F.leaky_relu(out)
        out = self.BN2(out)
        out = self.mp2d(out)

        out = self.conv3(out)
        out = F.leaky_relu(out)
        out = self.BN3(out)
        out = self.mp2d(out)

        out = self.conv33(out)
        out = F.leaky_relu(out)
        out = self.BN3(out)
        out = self.mp2d(out)

        out = self.conv4(out)
        out = F.leaky_relu(out)

        out = self.conv44(out)
        out = F.leaky_relu(out)
        out = self.BN4(out)
        out = self.mp2d(out)

        out = self.conv5(out)
        out = F.leaky_relu(out)

        out = self.conv55(out)
        out = F.leaky_relu(out)
        out = self.BN5(out)
        out = self.mp2d(out)

        # flatten for nns
        out = out.view(out.shape[0], -1)

        # dropout 0.5
        out = self.dp(out)
        out = self.fc1(out)
        out = F.leaky_relu(out)

        out = self.dp(out)
        out = self.fc2(out)
        out = F.leaky_relu(out)

        if self.feature: return out

        out = self.fc3(out)
        out = self.sm(out)

        return out


class RetResNet(nn.Module):
    def __init__(self):
        super(RetResNet, self).__init__()
        self.resnet = models.resnet18(True)
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.resnet.fc = nn.Linear(512, 256)
        self.fc = nn.Linear(256, 5)

    def forward(self, x):
        out = self.resnet.forward(x)
        out = F.leaky_relu(out)
        return self.fc(out)
