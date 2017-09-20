from __future__ import division
from __future__ import print_function

from caffe import layers as L, params as P, to_proto
from caffe.proto import caffe_pb2
import caffe


# Batch Normalization - PReLU - Convolution - [Dropout] layer
# used in Dense Block (DB) and Transition Down (TD).
# param:
#   bottom:  bottom blob
#   ks:      kernel_size used in convolution layer, [default] 3 in DB, 1 in TD
#   nout:    num_output used in convolution layer, be known also as "growth rate" in DB
#   stride:  stride used in convolution layer, [default] 1 in both DB and TD
#   pad:     pad used in convolution layer, [default] 1 in DB, 0 in TD
#   dropout: dropout_ratio used in dropout layer, [default] 0.2
def bn_relu_conv(net, flag, bottom, ks, nout, stride, pad, dropout):
  suffix     = '{}x{}'.format(ks, ks)
  flag_bn    = '{}_{}_bn'.format(flag, suffix)
  flag_scale = '{}_{}_scale'.format(flag, suffix)
  flag_relu  = '{}_{}_relu'.format(flag, suffix)
  flag_conv  = '{}_{}_conv'.format(flag, suffix)
  flag_drop  = '{}_{}_dropout'.format(flag, suffix)
  
  net[flag_bn] = L.BatchNorm(bottom, in_place=False,
                             batch_norm_param = dict(use_global_stats=True))
  net[flag_scale] = L.Scale(net[flag_bn], bias_term=True, in_place=True)
  net[flag_relu] = L.PReLU(net[flag_scale], in_place=True)
  
  net[flag_conv] = L.Convolution(net[flag_relu], num_output=nout, 
                                kernel_size=ks, stride=stride, pad=pad, 
                                bias_term=False)
  
  if dropout > 0:
    net[flag_drop] = L.Dropout(net[flag_conv], dropout_ratio=dropout)
    return net[flag_drop]
  
  return net[flag_conv]


# concat layer
# concat input and output blobs in the same bn_relu_conv layer, 
# in order to concat any layer to all subsequent layers in the same DB.
# param:
#   bottom:     bottom blob
#   num_filter: num_output used in convolution layer, be known also as "growth rate" in DB
#   dropout:    dropout_ratio
def cat_layer(net, major, minor, bottom, num_filter, dropout):
  flag_brc = 'block_{}_{}'.format(major, minor)
  flag_cat = 'block_{}_{}_concat'.format(major, minor)
  
  # convolution 1*1, [B] Bottleneck layer in DB
  bottleneck = bn_relu_conv(net, flag_brc, bottom, ks=1, nout=num_filter*4, 
                            stride=1, pad=0, dropout=dropout)
  # convolution 3*3
  brc_layer = bn_relu_conv(net, flag_brc, bottleneck, ks=3, nout=num_filter, 
                           stride=1, pad=1, dropout=dropout)
  
  net[flag_cat] = L.Concat(bottom, brc_layer, axis=1)
  
  return net[flag_cat]


# transition down
# reduce the spatial dimensionality via convolution and pooling.
# param:
#   bottom:     bottom blob
#   num_filter: num_output used in convolution layer
#   dropout:    dropout_ratio
def transition_down(net, major, bottom, num_filter, dropout):
  flag_brc  = 'transition_down_{}'.format(major)
  flag_pool = 'transition_down_{}_pooling'.format(major)
  
  # [C] 1/ratio < 1.0
  ratio = 2 # 1/ratio=0.5
  brc_layer = bn_relu_conv(net, flag_brc, bottom, ks=1, nout=num_filter//ratio, 
                           stride=1, pad=0, dropout=dropout)
  net[flag_pool] = L.Pooling(brc_layer, pool=P.Pooling.AVE, kernel_size=2, stride=2)
  
  return net[flag_pool]


# DenseNet Architecture
# param:
#   bs:          batch_size
#   nlayer:      list the number of bn_relu_conv layers in each DB
#   nclass:      the number of classes
#   first_nout:  num_output used in first convolution layer before entering the first DB, 
#                set it to be comparable to growth_rate
#   growth_rate: growth rate, in reference to num_output used in convolution layers in DB
#   dropout:     dropout_ratio, set to 0 to disable dropout
def densenet(nlayer, nclass, first_nout=16, growth_rate=16, dropout=0.2):

  net = caffe.NetSpec()
  
  net.data  = L.Input(input_param = dict(shape = dict(dim = [1,3,224,224])))

  pre_fmap = 0 # total number of previous feature maps
  
  # first convolution --------------------------------------------------------
  net.conv_1 = L.Convolution(net.data, num_output=first_nout, 
                             kernel_size=7, stride=2, pad=3)
  net.relu_1 = L.PReLU(net.conv_1, in_place=True)  
  net.pool_1 = L.Pooling(net.relu_1, pool=P.Pooling.MAX,
                         kernel_size=3, stride=2)
  
  pre_layer = net.pool_1
  pre_fmap += first_nout
  
  # DB + TD ------------------------------------------------------------------
  # +1 in order to make the index values from 1
  for major in xrange(len(nlayer)-1):
    # DB
    for minor in xrange(nlayer[major]):
      pre_layer = cat_layer(net, major+1, minor+1, pre_layer, growth_rate, dropout)
      pre_fmap += growth_rate
    # TD
    pre_layer = transition_down(net, major+1, pre_layer, pre_fmap, dropout)
    pre_fmap = pre_fmap // 2
  
  # last DB, without TD
  major = len(nlayer)
  for minor in xrange(nlayer[-1]):
    pre_layer = cat_layer(net, major, minor+1, pre_layer, growth_rate, dropout)
    pre_fmap += growth_rate
  
  # final layers -------------------------------------------------------------
  net.bn_final = L.BatchNorm(pre_layer, in_place=False, 
                             batch_norm_param = dict(use_global_stats=True))
  net.scale_finel = L.Scale(net.bn_final, bias_term=True, in_place=True)
  net.relu_final = L.PReLU(net.scale_finel, in_place=True)
  net.pool_final = L.Pooling(net.relu_final, pool=P.Pooling.AVE, global_pooling=True)
  
  net.fc_class = L.InnerProduct(net.pool_final, num_output=nclass)
  
  return str(net.to_proto())


def construct_net():
  # DenseNet-121(k=32)
  growth_rate = 32
  nlayer = [6,12,24,16]
  # DenseNet-169(k=32)
  #growth_rate = 32
  #nlayer = [6,12,32,32]
  # DenseNet-201(k=32)
  #growth_rate = 32
  #nlayer = [6,12,48,32]
  # DenseNet-161(k=48)
  #growth_rate = 48
  #nlayer = [6,12,36,24]
  
  first_nout  = growth_rate * 2
  nclass = 1000
  
  total_num_layer = sum(nlayer)*2 + 5
  file_name = 'densenet_{}_deploy.prototxt'.format(total_num_layer)
  net_name  = 'name: "DenseNet-{}_deploy"\n'.format(total_num_layer)
  net_arch = densenet(nlayer, nclass, first_nout=first_nout,growth_rate=growth_rate)
  with open(file_name, 'w') as f:
    f.write(net_name)
    f.write(net_arch)


if __name__ == '__main__':
  construct_net()
