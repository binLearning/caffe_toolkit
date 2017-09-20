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
  block_flag = 'block_{}_branch{}'.format(major, minor)
  bn_layer    = '{}_bn'.format(block_flag)
  scale_layer = '{}_scale'.format(block_flag)
  relu_layer  = '{}_relu'.format(block_flag)
  conv_layer  = '{}_conv'.format(block_flag)
  
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
def _block(flag, net, bottom, nout, has_branch1=False, increasing_dims=True, dropout=0.3):
  eltwise_layer = 'block_{}_addition'.format(flag)

  stride = 1
  if has_branch1 and increasing_dims:
    stride = 2

  branch2a = _block_4in1(flag, '2a', net, bottom,   nout, 1, 3, stride)
  if dropout > 0:
    dropout_layer = 'block_{}_dropout'.format(flag)
    net[dropout_layer] = L.Dropout(branch2a, dropout_ratio=dropout)
    branch2b = _block_4in1(flag, '2b', net, net[dropout_layer], nout, 1, 3, 1)
  else:
    branch2b = _block_4in1(flag, '2b', net, branch2a, nout, 1, 3, 1)

  if has_branch1:
    branch1 = _block_4in1(flag, '1', net, bottom, nout, 1, 3, stride)
    net[eltwise_layer] = L.Eltwise(branch1, branch2b)
  else:
    net[eltwise_layer] = L.Eltwise(bottom, branch2b)

  return net[eltwise_layer]


def construc_net(widening_factor, num_block_per_stage):
  net = caffe.NetSpec()

  net.data  = L.Input(input_param = dict(shape = dict(dim = [1,3,32,32])))

  net.conv1 = L.Convolution(net.data, num_output = 16,
                            kernel_size = 3, stride = 1, pad = 1,
                            bias_term = False)

  # stage 1
  num_out = widening_factor * 16
  block_pre = _block('2_1', net, net.conv1, num_out, has_branch1=True, increasing_dims=False)
  for idx in xrange(2,num_block_per_stage+1,1):
    flag = '2_{}'.format(idx)
    block_pre = _block(flag, net, block_pre, num_out)

  # stage 2
  num_out = widening_factor * 32
  block_pre = _block('3_1', net, block_pre, num_out, has_branch1=True)
  for idx in xrange(2,num_block_per_stage+1,1):
    flag = '3_{}'.format(idx)
    block_pre = _block(flag, net, block_pre, num_out)
  
  # stage 3
  num_out = widening_factor * 64
  block_pre = _block('4_1', net, block_pre, num_out, has_branch1=True)
  for idx in xrange(2,num_block_per_stage+1,1):
    flag = '4_{}'.format(idx)
    block_pre = _block(flag, net, block_pre, num_out)

  net.bn5    = L.BatchNorm(block_pre)
  net.scale5 = L.Scale(net.bn5, bias_term = True, in_place=True)
  net.relu5  = L.ReLU(net.scale5, in_place = True)
  net.pool5  = L.Pooling(net.relu5, pool = P.Pooling.AVE, global_pooling=True)
  
  net.fc6 = L.InnerProduct(net.pool5, num_output = 10)
  net.prob = L.Softmax(net.fc6)

  return net.to_proto()


def main(args):
  num_block_per_stage = (args.depth - 4) // 6
  
  file_name = 'wrn_{}_{}_deploy.prototxt'.format(args.depth, args.wfactor)
  net_name  = 'name: "WRN-{}-{}_deploy"\n'.format(args.depth, args.wfactor)
  
  with open(file_name, 'w') as f:
    f.write(net_name)
    f.write(str(construc_net(args.wfactor, num_block_per_stage)))


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('depth', type = int,
                      help = 'depth should be 6n+4 (e.g., 16,22,28,40 in the paper)')
  parser.add_argument('wfactor', type = int,
                      help = ' widening factor k, multiplies the number of features in conv layers')
  args = parser.parse_args()
  
  main(args)
