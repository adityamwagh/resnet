import torch.nn as nn
import torch.nn.functional as F
from frelu import FReLU

class BasicBlock(nn.Module):

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_channels[0]

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( num_channels[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def Resnet18(num_of_block, num_of_channel):
    return ResNet(BasicBlock, num_of_block, num_of_channel)

######## Model2 with leaky relu
class BasicBlock_leaky(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_leaky, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out)
        return out


class ResNet_leaky(nn.Module):
    def __init__(self, block, num_blocks, num_channels, num_classes=10):
        super(ResNet_leaky, self).__init__()
        self.in_planes = num_channels[0]

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( num_channels[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def Resnet18_leaky(num_of_block, num_of_channel):
    return ResNet_leaky(BasicBlock_leaky, num_of_block, num_of_channel)


#######3############################################################
###### Implementiing Funnel Activation Function with --Resnet 18 --###################

class BasicBlock_frelu(nn.Module):
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_frelu, self).__init__()
        base_width =1 
        groups = 1
        width = int(planes * (base_width / 42.)) * groups
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.frelu1 = FReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.frelu2 = FReLU(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.frelu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.frelu2(out)
        return out
        
class ResNet_frelu(nn.Module):
    def __init__(self, block, num_blocks, num_channels, num_classes=10):
        super(ResNet_frelu, self).__init__()
        self.in_planes = num_channels[0]

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.frelu1 = FReLU(self.in_planes)
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], stride=2)
        self.linear = nn.Linear( num_channels[3], num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.frelu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
        
def Resnet18_frelu(num_of_block, num_of_channel):
    return ResNet_frelu(BasicBlock_frelu, num_of_block, num_of_channel)









