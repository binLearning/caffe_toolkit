from __future__ import division

import os
import argparse
import numpy as np
from six.moves import xrange

import caffe
from caffe import layers as L, params as P
from caffe import to_proto


# 4(layer) in 1(block)
# BatchNorm - Scale - ReLU - Convolution
def _block_4in1(major, minor, net, bottom, nout, pad, ks, stride):
  block_flag = 'block{}_branch{}'.format(major, minor)
  bn_layer    = 'bn_{}'.format(block_flag)
  scale_layer = 'scale_{}'.format(block_flag)
  relu_layer  = 'relu_{}'.format(block_flag)
  conv_layer  = 'conv_{}'.format(block_flag)
  
  net[bn_layer]    = L.BatchNorm(bottom)
  net[scale_layer] = L.Scale(net[bn_layer], bias_term = True, in_place = True)
  net[relu_layer]  = L.ReLU(net[scale_layer], in_place = True)
  net[conv_layer]  = L.Convolution(net[relu_layer],
                                   num_output = nout, pad = pad,
                                   kernel_size = ks, stride = stride,
                                   bias_term = False)

  return net[conv_layer]


# block (residual unit)
#              [4in1] \             for increasing dimensions (decreasing spatial dimensions)
#                      | - block
# 4in1 - 4in1 - 4in1  /
def _block(major, net, bottom, nout, has_block1=False, increasing_dims=True):
  eltwise_layer = 'addition_block{}'.format(major)

  stride = 1
  if has_block1 and increasing_dims:
    stride = 2

  branch2a = _block_4in1(major, '2a', net, bottom,   nout//4, 0, 1, stride)
  branch2b = _block_4in1(major, '2b', net, branch2a, nout//4, 1, 3, 1)
  branch2c = _block_4in1(major, '2c', net, branch2b, nout,    0, 1, 1)

  if has_block1:
    branch1 = _block_4in1(major, '1', net, bottom, nout, 0, 1, stride)
    net[eltwise_layer] = L.Eltwise(branch1, branch2c)
  else:
    net[eltwise_layer] = L.Eltwise(bottom, branch2c)

  return net[eltwise_layer]


def construc_net(num_block_per_stage):
  net = caffe.NetSpec()

  net.data  = L.Input(input_param = dict(shape = dict(dim = [1,3,32,32])))

  net.conv1 = L.Convolution(net.data, num_output = 16,
                            kernel_size = 3, stride = 1, pad = 1,
                            bias_term = False)

  # stage 1
  block_pre = _block('2_1', net, net.conv1, 64, has_block1 = True, increasing_dims = False)
  for idx in xrange(2,num_block_per_stage+1,1):
    flag = '2_{}'.format(idx)
    block_pre = _block(flag, net, block_pre, 64)

  # stage 2
  block_pre = _block('3_1', net, block_pre, 128, has_block1 = True)
  for idx in xrange(2,num_block_per_stage+1,1):
    flag = '3_{}'.format(idx)
    block_pre = _block(flag, net, block_pre, 128)
  
  # stage 3
  block_pre = _block('4_1', net, block_pre, 256, has_block1 = True)
  for idx in xrange(2,num_block_per_stage+1,1):
    flag = '4_{}'.format(idx)
    block_pre = _block(flag, net, block_pre, 256)

  net.bn5    = L.BatchNorm(block_pre)
  net.scale5 = L.Scale(net.bn5, bias_term = True, in_place = True)
  net.relu5  = L.ReLU(net.scale5, in_place = True)
  net.pool5  = L.Pooling(net.relu5, pool = P.Pooling.AVE, kernel_size = 8, stride = 1)
  
  net.fc6 = L.InnerProduct(net.pool5, num_output = 10)
  net.prob = L.Softmax(net.fc6)

  return net.to_proto()


def main(args):
  num_block_per_stage = (args.depth - 2) // 9
  
  file_name = 'resnet_v2_{}_deploy.prototxt'.format(args.depth)
  net_name  = 'name: "ResNet-v2-{}_deploy"\n'.format(args.depth)
  
  with open(file_name, 'w') as f:
    f.write(net_name)
    f.write(str(construc_net(num_block_per_stage)))

    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('depth', type = int,
                      help = 'depth should be 9n+2 (e.g., 164 or 1001 in the paper)')
  args = parser.parse_args()
  
  main(args)
