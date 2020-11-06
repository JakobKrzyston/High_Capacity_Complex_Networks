"""
# Author
Jakob Krzyston (jakobk@gatech.edu)

# Purpose
Build DenseNet architecture for I/Q modulation classification
Code adapted from original Pytorch DenseNet code (https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List


##### LINEAR COMBINATION FOR COMPLEX CONVOLUTION #####

class LC(nn.Module):
    def __init__(self):
        super(LC, self).__init__()
        #this matrix adds the first and third columns of the output of Conv2d
    def forward(self, x):
        i = x[:,:,0:1,:]-x[:,:,2:3,:]
        q = x[:,:,1:2,:]
        return torch.cat([i,q],dim=2)


##### RESIDUAL LAYERS #####
class _ResLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, kernel_sz, memory_efficient=False):
        super(_ResLayer, self).__init__()
        if kernel_sz == 3:
            pad = 1
        elif kernel_sz == 5:
            pad = 2
        elif kernel_sz == 7:
            pad = 3
        
        out_planes = bn_size * growth_rate

        # Conv and BN for the bn_function, as seen in DenseNet
        self.conv_bn = nn.Conv2d(num_input_features, bn_size * growth_rate, \
        						kernel_size=(1,3), stride=1, padding = (0,1), bias=False)
        self.bn_bn   = nn.BatchNorm2d(num_input_features)
        

        # Layers for 4 Residual Units as seen in Liu et al. 2020
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes,out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv4 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn4 = nn.BatchNorm2d(out_planes)
        self.conv5 = nn.Conv2d(out_planes,out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn5 = nn.BatchNorm2d(out_planes)
        self.conv6 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn6 = nn.BatchNorm2d(out_planes)
        self.conv7 = nn.Conv2d(out_planes,out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn7 = nn.BatchNorm2d(out_planes)
        self.conv8 = nn.Conv2d(out_planes, out_planes, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn8 = nn.BatchNorm2d(out_planes)
        
        # Extra Conv layer for proper number of channels after the last residual connection,
        # for the Transition Layer
        self.conv9 = nn.Conv2d(out_planes, growth_rate, kernel_size=(1,kernel_sz), stride=1,\
                               padding=(0,pad), bias=False)
        self.bn9 = nn.BatchNorm2d(growth_rate)
        
        self.relu = nn.ReLU(inplace=True)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv_bn(self.relu(self.bn_bn(concated_features)))  # noqa: T484
        return bottleneck_output

    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def forward(self, input):  
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        x = self.bn_function(prev_features)

        # Residual Units
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)

        identity2 = out
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.bn4(self.conv4(out))
        out += identity2
        out = self.relu(out)
        identity3 = out
        out = self.relu(self.bn5(self.conv5(out)))
        out = self.bn6(self.conv6(out))
        out += identity3
        out = self.relu(out)
        identity4 = out
        out = self.relu(self.bn7(self.conv7(out)))
        out = self.bn8(self.conv8(out))
        out += identity4
        out = self.relu(out)
        out = self.relu(self.bn9(self.conv9(out)))
        return out


class _ResLayer_c(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, kernel_sz, memory_efficient=False):
        super(_ResLayer_c, self).__init__()
        if kernel_sz == 3:
            pad = 1
        elif kernel_sz == 5:
            pad = 2
        elif kernel_sz == 7:
            pad = 3
        
        out_planes = bn_size * growth_rate

        # Conv and BN for the bn_function, as seen in DenseNet
        self.conv_bn = nn.Conv2d(num_input_features, bn_size * growth_rate, \
                                kernel_size=(2,3), stride=1, padding = (1,1), bias=False)
        self.bn_bn   = nn.BatchNorm2d(num_input_features)
        

        # Layers for 4 Residual Units as seen in Liu et al. 2020
        self.conv1 = nn.Conv2d(out_planes, out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv3 = nn.Conv2d(out_planes,out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.conv4 = nn.Conv2d(out_planes, out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn4 = nn.BatchNorm2d(out_planes)
        self.conv5 = nn.Conv2d(out_planes,out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn5 = nn.BatchNorm2d(out_planes)
        self.conv6 = nn.Conv2d(out_planes, out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn6 = nn.BatchNorm2d(out_planes)
        self.conv7 = nn.Conv2d(out_planes,out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn7 = nn.BatchNorm2d(out_planes)
        self.conv8 = nn.Conv2d(out_planes, out_planes, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn8 = nn.BatchNorm2d(out_planes)
        
        # Extra Conv layer for proper number of channels after the last residual connection,
        # for the Transition Layer
        self.conv9 = nn.Conv2d(out_planes, growth_rate, kernel_size=(2,kernel_sz), stride=1,\
                               padding=(1,pad), bias=False)
        self.bn9 = nn.BatchNorm2d(growth_rate)
        
        self.relu = nn.ReLU(inplace=True)

    def bn_function(self, inputs):
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = LC.forward(self,(self.conv_bn(self.relu(self.bn_bn(concated_features)))))  # noqa: T484
        return bottleneck_output

    def any_requires_grad(self, input):
        # type: (List[Tensor]) -> bool
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    def forward(self, input):  
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        x = self.bn_function(prev_features)

        # Residual Units
        identity = x
        out = self.relu(self.bn1(LC.forward(self,(self.conv1(x)))))
        out = self.bn2(LC.forward(self,(self.conv2(out))))
        out += identity
        out = self.relu(out)

        identity2 = out
        out = self.relu(self.bn3(LC.forward(self,(self.conv3(out)))))
        out = self.bn4(LC.forward(self,(self.conv4(out))))
        out += identity2
        out = self.relu(out)
        identity3 = out
        out = self.relu(self.bn5(LC.forward(self,(self.conv5(out)))))
        out = self.bn6(LC.forward(self,(self.conv6(out))))
        out += identity3
        out = self.relu(out)
        identity4 = out
        out = self.relu(self.bn7(LC.forward(self,(self.conv7(out)))))
        out = self.bn8(LC.forward(self,(self.conv8(out))))
        out += identity4
        out = self.relu(out)
        out = self.relu(self.bn9(LC.forward(self,(self.conv9(out)))))
        return out


##### DENSE BLOCKS #####
class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, kernel_sz, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _ResLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                kernel_sz = kernel_sz,
            )
            self.add_module('reslayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _DenseBlock_c(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, kernel_sz, memory_efficient=False):
        super(_DenseBlock_c, self).__init__()
        for i in range(num_layers):
            layer = _ResLayer_c(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                kernel_sz = kernel_sz,
            )
            self.add_module('reslayer_c%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

    
##### TRANSITION BLOCKS #####

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          padding = (0,3), kernel_size=(1,7), stride=1,
                                          bias=False))

class _Transition_c(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition_c, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          padding = (1,3), kernel_size=(2,7), stride=1,
                                          bias=False))
        self.add_module('LC', LC())


##### DENSENETS #####

class DenseResNet(nn.Module):
    r"""
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate, block_config,num_init_features, kernel_sz,
                 bn_size=4, drop_rate=0, num_classes=11, memory_efficient=False):
        super(DenseResNet, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=(1,3), stride=1,
                                padding=(0,1), bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                kernel_sz=kernel_sz[i],
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Dense layers seen in Liu et al. 2020
        self.fc1     = nn.Linear(num_features, 128)
        self.fc2     = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p = 0.5)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out

class DenseResNet_c(nn.Module):
    r"""
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate, block_config,num_init_features, kernel_sz,
                 bn_size=4, drop_rate=0, num_classes=11, memory_efficient=False):

        super(DenseResNet_c, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(1, num_init_features, kernel_size=(2,3), stride=1, padding=(1,1), bias=False)),
            ('LC', LC()),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock_c(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
                kernel_sz = kernel_sz[i],
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition_c(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Dense layers seen in Liu et al. 2020
        self.fc1     = nn.Linear(num_features, 128)
        self.fc2     = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(p = 0.5)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.dropout(self.fc1(out))
        out = self.fc2(out)
        return out


##### FUNCTIONS TO BUILD ARCHITECTURE #####

def _denseresnet(arch, growth_rate, block_config, num_init_features, Complex, kernel_sz, **kwargs):
    if Complex != True:
        model = DenseResNet(growth_rate, block_config, num_init_features, kernel_sz, **kwargs)
    else:
        model = DenseResNet_c(growth_rate, block_config, num_init_features, kernel_sz, **kwargs)
    return model


def denseresnet35(Complex = False, **kwargs):
    """DenseResnet-35"""
    return _denseresnet('denseresnet35', 32, (1,1,1), 64, Complex, kernel_sz = (7,5,3), **kwargs)

def denseresnet35_c(Complex = True, **kwargs):
    """DenseResnet-35 C"""
    return _denseresnet('denseresnet35_c', 32, (1,1,1), 64, Complex, kernel_sz = (7,5,3), **kwargs)

def denseresnet68(Complex = False, **kwargs):
    """DenseResnet-68"""
    return _denseresnet('denseresnet68', 32, (2,2,2), 64, Complex, kernel_sz = (7,5,3), **kwargs)

def denseresnet68_c(Complex = True, **kwargs):
    """DenseResnet-68 C"""
    return _denseresnet('denseresnet68_c', 32, (2,2,2), 64, Complex, kernel_sz = (7,5,3), **kwargs)