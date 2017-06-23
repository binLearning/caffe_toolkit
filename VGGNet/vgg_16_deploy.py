import os
import numpy as np
from six.moves import xrange

import caffe
from caffe import layers as L, params as P
from caffe import to_proto


# [Convolution - ReLU] * #stack - Pooling
def _block_crp(major, stack_num, net, bottom, nout, pad=1, ks=3, stride=1):
  for minor in xrange(stack_num):
    conv_layer = 'conv{}_{}'.format(major, minor+1)
    relu_layer = 'relu{}_{}'.format(major, minor+1)

    if minor == 0:
      bottom_layer = bottom
    else:
      pre_layer = 'relu{}_{}'.format(major, minor)
      bottom_layer = net[pre_layer]

    net[conv_layer] = L.Convolution(bottom_layer,
                                    num_output = nout, pad = pad,
                                    kernel_size = ks, stride = stride)
    net[relu_layer] = L.ReLU(net[conv_layer], in_place = True)

  pool_layer = 'pool{}'.format(major)
  net[pool_layer] = L.Pooling(net[relu_layer], pool = P.Pooling.MAX, 
                              kernel_size = 2, stride = 2)

  return net[pool_layer]


# FullyConnection - ReLU - Dropout
def _block_frd(major, net, bottom, nout, dropratio=0.5):
  fc_layer = 'fc{}'.format(major)
  relu_layer = 'relu{}'.format(major)
  drop_layer = 'drop{}'.format(major)

  net[fc_layer] = L.InnerProduct(bottom, num_output = nout)
  net[relu_layer] = L.ReLU(net[fc_layer], in_place = True)
  net[drop_layer] = L.Dropout(net[relu_layer], dropout_ratio = dropratio,
                              in_place = True)
  
  return net[drop_layer]


def construc_net():
  net = caffe.NetSpec()

  net.data = L.Input(shape = dict(dim = [10,3,224,224]))

  block_1 = _block_crp('1', 2, net, net.data, 64)
  block_2 = _block_crp('2', 2, net, block_1,  128)
  block_3 = _block_crp('3', 3, net, block_2,  256)
  block_4 = _block_crp('4', 3, net, block_3,  512)
  block_5 = _block_crp('5', 3, net, block_4,  512)
  
  block_6 = _block_frd('6', net, block_5, 4096)
  block_7 = _block_frd('7', net, block_6, 4096)

  net.fc8 = L.InnerProduct(block_7, num_output = 1000)
  net.prob = L.Softmax(net.fc8)

  return net.to_proto()


def main():
  with open('vgg_16_deploy.prototxt', 'w') as f:
    f.write('name: "VGG-16_deploy"\n')
    f.write(str(construc_net()))

    
if __name__ == '__main__':
  main()
