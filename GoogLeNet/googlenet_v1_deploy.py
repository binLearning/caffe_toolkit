import os
import numpy as np
from six.moves import xrange

import caffe
from caffe import layers as L, params as P
from caffe import to_proto


# block
# Convolution - ReLU
def _block_cr(major, minor, net, bottom, nout, pad, ks, stride):
  conv_layer = '{}/{}'.format(major, minor)
  relu_layer = '{}/relu_{}'.format(major, minor)

  net[conv_layer] = L.Convolution(bottom,
                                  num_output = nout, pad = pad,
                                  kernel_size = ks, stride = stride)
  net[relu_layer] = L.ReLU(net[conv_layer], in_place = True)

  return net[relu_layer]


# Inception v1
def _inception_v1(major, net, bottom, nout):
  minor = ['1x1', '3x3_reduce', '3x3 ', '5x5_reduce', '5x5', 'pool_proj']

  block_1x1 = _block_cr(major, minor[0], net, bottom, nout[0], 0, 1, 1)

  block_3x3_reduce = _block_cr(major, minor[1], net, bottom, nout[1], 0, 1, 1)
  block_3x3 = _block_cr(major, minor[2], net, block_3x3_reduce, nout[2], 1, 3, 1)

  block_5x5_reduce = _block_cr(major, minor[3], net, bottom, nout[3], 0, 1, 1)
  block_5x5 = _block_cr(major, minor[4], net, block_5x5_reduce, nout[4], 2, 5, 1)

  pool_layer = '{}/pool'.format(major)
  net[pool_layer] = L.Pooling(bottom, pool = P.Pooling.MAX,
                         pad = 1, kernel_size = 3, stride = 1)
  block_pool_proj = _block_cr(major, minor[5], net, net[pool_layer], nout[5], 0, 1, 1)

  output_layer = '{}/output'.format(major)
  net[output_layer] = L.concat(block_1x1, block_3x3, block_5x5, block_pool_proj)

  return net[output_layer]


def construc_net():
  net = caffe.NetSpec()

  net.data = L.Input(shape = dict(dim = [10,3,224,224]))
  block_cr_1 = _block_cr('conv1', '7x7_s2', net, net.data, 64, 3, 7, 2)
  pool_layer_1 = 'pool1/3x3_s2'
  net[pool_layer_1] = L.Pooling(block_cr_1, pool = P.Pooling.MAX,
                                kernel_size = 3, stride = 2)
  ##LRN
  block_cr_2_reduce = _block_cr('conv2', '3x3_reduce', net, net[pool_layer_1], 64, 0, 1, 1)
  block_cr_2 = _block_cr('conv2', '3x3', net, block_cr_2_reduce, 192, 1, 3, 1)
  ##LRN
  pool_layer_2 = 'pool2/3x3_s2'
  net[pool_layer_2] = L.Pooling(block_cr_2, pool = P.Pooling.MAX,
                                kernel_size = 3, stride = 2)
  inception_3a = _inception_v1('inception_3a', net, net[pool_layer_2], [64,96,128,16,32,32])
  inception_3b = _inception_v1('inception_3b', net, inception_3a, [128,128,192,32,96,64])
  pool_layer_3 = 'pool3/3x3_s2'
  net[pool_layer_3] = L.Pooling(inception_3b, pool = P.Pooling.MAX,
                                kernel_size = 3, stride = 2)
  inception_4a = _inception_v1('inception_4a', net, net[pool_layer_3], [192,96,208,16,48,64])
  inception_4b = _inception_v1('inception_4b', net, inception_4a, [160,112,224,24,64,64])
  inception_4c = _inception_v1('inception_4c', net, inception_4b, [128,128,256,24,64,64])
  inception_4d = _inception_v1('inception_4d', net, inception_4c, [112,144,288,32,64,64])
  inception_4e = _inception_v1('inception_4e', net, inception_4d, [256,160,320,32,128,128])
  pool_layer_4 = 'pool4/3x3_s2'
  net[pool_layer_4] = L.Pooling(inception_4e, pool = P.Pooling.MAX,
                                kernel_size = 3, stride = 2)
  inception_5a = _inception_v1('inception_5a', net, net[pool_layer_4], [256,160,320,32,128,128])
  inception_5b = _inception_v1('inception_5b', net, inception_5a, [384,192,384,48,128,128])
  pool_layer_5 = 'pool5/7x7_s1'
  net[pool_layer_5] = L.Pooling(inception_5b, pool = P.Pooling.AVE,
                                kernel_size = 7, stride = 1)
  pool_layer_5_drop = 'pool5/drop_7x7_s1'
  net[pool_layer_5_drop] = L.Dropout(net[pool_layer_5], dropout_ratio = 0.4, in_place = True)
  fc_layer = 'loos3/classifier'
  net[fc_layer] = L.InnerProduct(net[pool_layer_5_drop], num_output = 1000)
  net.prob = L.Softmax(net[fc_layer])

  return net.to_proto()


def main():
  with open('googlenet_v1_deploy.prototxt', 'w') as f:
    f.write('name: "GoogLeNet-v1_deploy"\n')
    f.write(str(construc_net()))

if __name__ == '__main__':
  main()
