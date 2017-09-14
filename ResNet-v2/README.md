# ResNet-v2
[Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027) <br>
Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun <br>

### 摘要
近期已经涌现出很多以深度残差网络（deep residual network）为基础的极深层的网络架构，在准确率和收敛性等方面的表现都非常引人注目。
本文主要分析残差网络基本构件（block）中的信号传播，我们发现当使用恒等映射（identity mapping）作为快捷连接（skip connection）
并且将激活函数移至加法操作后面时，前向-反向信号都可以在两个block之间直接传播而不受到任何变换操作的影响。大量实验结果证明了恒等映射的重要性。
本文根据这个发现重新设计了一种残差网络基本单元（unit），使得网络更易于训练并且泛化性能也得到提升。官方实现（Torch）的源码地址：
https://github.com/KaimingHe/resnet-1k-layers。 <br>

### Introduction
深度残差网络（ResNet）由“残差单元（Residual Units）”堆叠而成，每个单元可以表示为：









































































































