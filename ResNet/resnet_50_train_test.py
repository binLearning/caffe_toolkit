import os
import numpy as np

import caffe
from caffe import layers as L, params as P
from caffe import to_proto


# filler
# type:  'constant', 'gaussian', 'xavier'
#        [default: 'xavier']
# value: the value in constant filler
#        [default: 0.2]
# std:   the std value in Gaussian filler
#        [default: 0.01]
def _filler(filler_type='xavier', filler_value=0.2, filler_std=0.01):
  filler = {}

  if filler_type == 'xavier':
    filler = dict(type = filler_type)
  elif filler_type == 'gaussian':
    filler = dict(type = filler_type, std = filler_std)
  elif filler_type == 'constant':
    filler = dict(type = filler_type, value = filler_value)

  return filler  


# first block
# Convolution - BatchNorm - Scale - ReLU
def _block_first(net, bottom, nout=64, pad=3, ks=7, stride=2):
  net.conv1 = L.Convolution(bottom,
                            num_output = nout, pad = pad,
                            kernel_size = ks, stride = stride,
                            weight_filler = _filler(),
                            bias_filler = _filler(filler_type = 'constant'))
  net.bn_conv1 = L.BatchNorm(net.conv1, in_place = True)
  net.scale_conv1 = L.Scale(net.bn_conv1, bias_term = True, in_place = True)
  net.conv1_relu = L.ReLU(net.scale_conv1, in_place = True)

  return net.conv1_relu


# 3(layer) in 1(block)
# Convolution - BatchNorm - Scale
def _block_3in1(major, minor, net, bottom, nout, pad, ks, stride):
  branch_flag = '{}_branch{}'.format(major, minor)
  conv_layer  = 'res{}'.format(branch_flag)
  bn_layer    = 'bn{}'.format(branch_flag)
  scale_layer = 'scale{}'.format(branch_flag)

  net[conv_layer]  = L.Convolution(bottom,
                                   num_output = nout, pad = pad,
                                   kernel_size = ks, stride = stride,
                                   weight_filler = _filler(),
                                   bias_filler = _filler(filler_type = 'constant'))
  net[bn_layer]    = L.BatchNorm(net[conv_layer], in_place = True)
  net[scale_layer] = L.Scale(net[bn_layer], bias_term = True, in_place = True)

  return net[scale_layer]


# 4(layer) in 1(block)
# Convolution - BatchNorm - Scale - ReLU
def _block_4in1(major, minor, net, bottom, nout, pad, ks, stride):
  branch_flag = '{}_branch{}'.format(major, minor)
  conv_layer  = 'res{}'.format(branch_flag)
  bn_layer    = 'bn{}'.format(branch_flag)
  scale_layer = 'scale{}'.format(branch_flag)
  relu_layer  = 'res{}_relu'.format(branch_flag)
  
  net[conv_layer]  = L.Convolution(bottom,
                                   num_output = nout, pad = pad,
                                   kernel_size = ks, stride = stride,
                                   weight_filler = _filler(),
                                   bias_filler = _filler(filler_type = 'constant'))
  net[bn_layer]    = L.BatchNorm(net[conv_layer], in_place = True)
  net[scale_layer] = L.Scale(net[bn_layer], bias_term = True, in_place = True)
  net[relu_layer]  = L.ReLU(net[scale_layer], in_place = True)

  return net[relu_layer]


# branch
#              [3in1] \  
#                      | - branch
# 4in1 - 4in1 - 3in1  /
def _branch(major, net, bottom, nout, has_branch1=False, is_branch_2a=False):
  eltwise_layer = 'res{}'.format(major)
  relu_layer    = 'res{}_relu'.format(major)

  stride = 1
  if has_branch1 and not is_branch_2a:
    stride = 2

  branch2_2a = _block_4in1(major, '2a', net, bottom, nout,       0, 1, stride)
  branch2_2b = _block_4in1(major, '2b', net, branch2_2a, nout,   1, 3, 1)
  branch2_2c = _block_3in1(major, '2c', net, branch2_2b, nout*4, 0, 1, 1)

  if has_branch1:
    branch1 = _block_3in1(major, '1',  net, bottom, nout*4, 0, 1, stride)
    net[eltwise_layer] = L.Eltwise(branch1, branch2_2c)
  else:
    net[eltwise_layer] = L.Eltwise(bottom, branch2_2c)

  net[relu_layer] = L.ReLU(net[eltwise_layer], in_place = True)

  return net[relu_layer]


def construc_net():
  net = caffe.NetSpec()

  _transform_train = dict(scale = 0.0078125, # 1/128
                          mirror = True,
                          #crop_size = 224,
                          mean_value = [127.5, 127.5, 127.5])
  _transform_test  = dict(scale = 0.0078125, # 1/128
                          mirror = False,
                          #crop_size = 224,
                          mean_value = [127.5, 127.5, 127.5])

  net.data, net.label = L.ImageData(include = dict(phase = 0), # TRAIN = 0,in caffe_pb2.py
                                    transform_param = _transform_train,
                                    source = '../data/images_train.txt',
                                    batch_size = 32,
                                    shuffle = True,
                                    #new_height = 224,
                                    #new_width = 224,
                                    #is_color = True,
                                    ntop = 2)
  
  # NOTE
  data_layer_train = net.to_proto()

  net.data, net.label = L.ImageData(include = dict(phase = 1), # TEST = 1
                                    transform_param = _transform_test,
                                    source = '../data/images_test.txt',
                                    batch_size = 4,
                                    shuffle = False,
                                    #new_height = 224,
                                    #new_width = 224,
                                    #is_color = True,
                                    ntop = 2)

  block1 = _block_first(net, net.data)

  net.pool1 = L.Pooling(block1, pool = P.Pooling.MAX, kernel_size = 3, stride = 2)

  branch_2a = _branch('2a', net, net.pool1, 64, has_branch1 = True, is_branch_2a = True)
  branch_2b = _branch('2b', net, branch_2a, 64)
  branch_2c = _branch('2c', net, branch_2b, 64)

  branch_3a = _branch('3a', net, branch_2c, 128, has_branch1 = True)
  branch_3b = _branch('3b', net, branch_3a, 128)
  branch_3c = _branch('3c', net, branch_3b, 128)
  branch_3d = _branch('3d', net, branch_3c, 128)

  branch_4a = _branch('4a', net, branch_3d, 256, has_branch1 = True)
  branch_4b = _branch('4b', net, branch_4a, 256)
  branch_4c = _branch('4c', net, branch_4b, 256)
  branch_4d = _branch('4d', net, branch_4c, 256)
  branch_4e = _branch('4e', net, branch_4d, 256)
  branch_4f = _branch('4f', net, branch_4e, 256)

  branch_5a = _branch('5a', net, branch_4f, 512, has_branch1 = True)
  branch_5b = _branch('5b', net, branch_5a, 512)
  branch_5c = _branch('5c', net, branch_5b, 512)

  net.pool5 = L.Pooling(branch_5c, pool = P.Pooling.AVE, kernel_size = 7, stride = 1)
  
  net.fc6 = L.InnerProduct(net.pool5, num_output = 1000)

  net.loss = L.SoftmaxWithLoss(net.fc6, net.label)

  net.accuracy = L.Accuracy(net.fc6, net.label, include = dict(phase = 1))

  return str(data_layer_train) + str(net.to_proto())


def main():
  file_name = 'resnet_50_train_test.prototxt'
  with open(file_name, 'w') as f:
    f.write('name: "ResNet-50_train_test"\n')
    f.write(construc_net())


if __name__ == '__main__':
  main()
