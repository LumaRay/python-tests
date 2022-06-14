import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class KitModel(nn.Module):
    '''class SeparableConv2d(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, bias=False):
            super(KitModel.SeparableConv2d, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                       groups=in_channels, bias=bias, padding=1)
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        def forward(self, x):
            out = self.depthwise(x)
            out = self.pointwise(out)
            return out'''

    def __init__(self, weight_file):
        super(KitModel, self).__init__()
        global _weights_dict
        _weights_dict = load_weights(weight_file)

        self.conv2d = self.__conv(2, name='conv2d', in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization = self.__batch_normalization(2, 'batch_normalization', num_features=32, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_1 = self.__conv(2, name='conv2d_1', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=True)
        self.batch_normalization_1 = self.__batch_normalization(2, 'batch_normalization_1', num_features=64, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_2 = self.__conv(2, name='conv2d_2', in_channels=64, out_channels=128, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_2 = self.__batch_normalization(2, 'batch_normalization_2', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_3 = self.__batch_normalization(2, 'batch_normalization_3', num_features=128, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_3 = self.__conv(2, name='conv2d_3', in_channels=128, out_channels=256, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_4 = self.__batch_normalization(2, 'batch_normalization_4', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_5 = self.__batch_normalization(2, 'batch_normalization_5', num_features=256, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_4 = self.__conv(2, name='conv2d_4', in_channels=256, out_channels=512, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_6 = self.__batch_normalization(2, 'batch_normalization_6', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_7 = self.__batch_normalization(2, 'batch_normalization_7', num_features=512, eps=0.0010000000474974513, momentum=0.0)
        self.conv2d_5 = self.__conv(2, name='conv2d_5', in_channels=512, out_channels=728, kernel_size=(1, 1), stride=(2, 2), groups=1, bias=True)
        self.batch_normalization_8 = self.__batch_normalization(2, 'batch_normalization_8', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_9 = self.__batch_normalization(2, 'batch_normalization_9', num_features=728, eps=0.0010000000474974513, momentum=0.0)
        self.batch_normalization_10 = self.__batch_normalization(2, 'batch_normalization_10', num_features=1024, eps=0.0010000000474974513, momentum=0.0)
        self.dense = self.__dense(name = 'dense', in_features = 1024, out_features = 5, bias = True)
        self.separable_conv2d = self.__separable_conv(2, name='separable_conv2d', in_channels=64, out_channels=128, kernel_size=(3, 3))  # , stride=(2, 2), groups=1, bias=True)

    def forward(self, x):
        # self.separable_conv = self.SeparableConv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))
        # self.separable_conv2d = self.__separable_conv(2, name='separable_conv2d', in_channels=64, out_channels=128, kernel_size=(3, 3))  # , stride=(2, 2), groups=1, bias=True)

        conv2d_pad      = F.pad(x, (0, 1, 0, 1))
        conv2d          = self.conv2d(conv2d_pad)
        batch_normalization = self.batch_normalization(conv2d)
        activation      = F.relu(batch_normalization)
        conv2d_1_pad    = F.pad(activation, (1, 1, 1, 1))
        conv2d_1        = self.conv2d_1(conv2d_1_pad)
        batch_normalization_1 = self.batch_normalization_1(conv2d_1)
        activation_1    = F.relu(batch_normalization_1) # saved
        activation_2    = F.relu(activation_1)
        # separable_conv2d = self.separable_conv2d(activation_2)
        separable_conv2d = self.__separable_conv(2, name='separable_conv2d', in_channels=64, out_channels=128, kernel_size=(3, 3), input=activation_2)
        conv2d_2        = self.conv2d_2(activation_1)
        batch_normalization_2 = self.batch_normalization_2(separable_conv2d)
        activation_3    = F.relu(batch_normalization_2)
        separable_conv2d_1 = self.separable_conv(activation_3)
        batch_normalization_3 = self.batch_normalization_3(separable_conv2d_1)
        max_pooling2d_pad = F.pad(batch_normalization_3, (0, 1, 0, 1), value=float('-inf'))
        max_pooling2d, max_pooling2d_idx = F.max_pool2d(max_pooling2d_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add             = max_pooling2d + conv2d_2
        activation_4    = F.relu(add)
        separable_conv2d_2 = self.separable_conv(activation_4)
        conv2d_3        = self.conv2d_3(add)
        batch_normalization_4 = self.batch_normalization_4(separable_conv2d_2)
        activation_5    = F.relu(batch_normalization_4)
        separable_conv2d_3 = self.separable_conv(activation_5)
        batch_normalization_5 = self.batch_normalization_5(separable_conv2d_3)
        max_pooling2d_1_pad = F.pad(batch_normalization_5, (0, 1, 0, 1), value=float('-inf'))
        max_pooling2d_1, max_pooling2d_1_idx = F.max_pool2d(max_pooling2d_1_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_1           = max_pooling2d_1 + conv2d_3
        activation_6    = F.relu(add_1)
        separable_conv2d_4 = self.separable_conv(activation_6)
        conv2d_4        = self.conv2d_4(add_1)
        batch_normalization_6 = self.batch_normalization_6(separable_conv2d_4)
        activation_7    = F.relu(batch_normalization_6)
        separable_conv2d_5 = self.separable_conv(activation_7)
        batch_normalization_7 = self.batch_normalization_7(separable_conv2d_5)
        max_pooling2d_2_pad = F.pad(batch_normalization_7, (0, 1, 0, 1), value=float('-inf'))
        max_pooling2d_2, max_pooling2d_2_idx = F.max_pool2d(max_pooling2d_2_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_2           = max_pooling2d_2 + conv2d_4
        activation_8    = F.relu(add_2)
        separable_conv2d_6 = self.separable_conv(activation_8)
        conv2d_5        = self.conv2d_5(add_2)
        batch_normalization_8 = self.batch_normalization_8(separable_conv2d_6)
        activation_9    = F.relu(batch_normalization_8)
        separable_conv2d_7 = self.separable_conv(activation_9)
        batch_normalization_9 = self.batch_normalization_9(separable_conv2d_7)
        max_pooling2d_3_pad = F.pad(batch_normalization_9, (0, 1, 0, 1), value=float('-inf'))
        max_pooling2d_3, max_pooling2d_3_idx = F.max_pool2d(max_pooling2d_3_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        add_3           = max_pooling2d_3 + conv2d_5
        separable_conv2d_8 = self.separable_conv(add_3)
        batch_normalization_10 = self.batch_normalization_10(separable_conv2d_8)
        activation_10   = F.relu(batch_normalization_10)
        global_average_pooling2d = F.avg_pool2d(input = activation_10, kernel_size = activation_10.size()[2:])
        global_average_pooling2d_flatten = global_average_pooling2d.view(global_average_pooling2d.size(0), -1)
        dropout         = F.dropout(input = global_average_pooling2d_flatten, p = 0.5, training = self.training, inplace = True)
        dense           = self.dense(dropout)
        dense_activation = F.softmax(dense)
        return dense_activation


    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    '''def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super(KitModel.SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out'''

    @staticmethod
    def __separable_conv(dim, name, in_channels, out_channels, kernel_size, bias=False, **kwargs):
        if   dim == 1:
            layer = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)(**kwargs)
            layer = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias)(layer)
        elif dim == 2:
            layer = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)(**kwargs).to(torch.device("cuda:0"), dtype=torch.half, non_blocking=True)
            layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)(layer)
        elif dim == 3:
            layer = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                                   groups=in_channels, bias=bias, padding=1)(**kwargs)
            layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=bias)(layer)
        else:           raise NotImplementedError()

        layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        if 'scale' in _weights_dict[name]:
            layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        else:
            layer.weight.data.fill_(1)

        if 'bias' in _weights_dict[name]:
            layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        else:
            layer.bias.data.fill_(0)

        layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

