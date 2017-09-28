# ShuffleNet
[ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices](https://arxiv.org/abs/1707.01083) <br>
Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun <br>

### 摘要
本文提出一种计算效率极高的CNN架构——ShuffleNet，主要应用于计算能力有限（例如10-150 MFLOPs）的移动设备中。ShuffleNet架构中利用了两个新的操作，
逐点分组卷积（pointwise group convolution）和通道随机重排（channel shuffle），在保持准确率的前提下极大地减少计算量。
在ImageNet分类和MS COCO检测任务上的实验表明，ShuffleNet的性能比其他结构（例如MobileNet）更优越。ShuffleNet在基于ARM的移动设备中的实际运行速度
要比AlexNet快约13倍，且准确率基本保持不变。 <br>

### 1. Introduction
本文主要聚焦于设计一种计算开支很小但准确率高的网络，主要应用于移动平台如无人机、机器人、手机等。之前的一些工作主要是对“基础”网络架构进行
修剪（pruning）、压缩（compressing）、低精度表示（low-bit representing）等处理来达到降低计算量的目的，而本文是要探索一种高效计算的基础网络。 <br>
当前最先进的基础网络架构如Xception、[ResNeXt](https://github.com/binLearning/caffe_toolkit/tree/master/ResNeXt)在极小的网络中计算效率变低，
主要耗费在密集的1x1卷积计算上。本文提出使用逐点分组卷积（pointwise group convolution）替代1x1卷积来减小计算复杂度，
另为了消除其带来的副作用，使用通道随机重排（channel shuffle）来改进特征通道中的信息流动，基于这两个操作构建了高效计算的ShuffleNet。
在同样的计算开支下，ShuffleNet比其他流行的机构有更多的特征图通道数量，可以编码更多的信息，这对于很小的网络的性能是尤为重要的。 <br>

### 2. Related Work
**Efficient Model Designs** <br>
在嵌入式设备上运行高质量深度神经网络的需求促进了对高效模型设计的研究。GoogLeNet在增加网络深度时随之增加的复杂度比简单堆叠卷积层的方式要低得多。
SqueezeNet可以在保持准确率的前提下大幅降低参数量和计算量。ResNet中使用bottleneck结构来提高计算效率。
AlexNet中提出分组卷积是为了将模型分配到两个GPU中，在ResNeXt中被发现可以用于提高网络性能。Xception中提出的深度可分离卷积
（depthwise separable convolution）是对Inception系列中可分离卷积思想的推广。MobileNet使用深度可分离卷积构建轻量级模型获得当前最先进的结果。
本文将分组卷积和深度可分离卷积推广到一种新的形式。 <br>
**Model Acceleration** <br>
这个方向主要是在保持预训练模型准确率的前提下对预测过程进行加速。有的方法是通过修剪网络连接或通道来减少预训练模型中的冗余连接。
量化（quantization）和因式分解（factorization）也可以减少冗余。还有一些方法并不是改变参数，而是用FFT或其他方法来优化卷积算法的实现以达到加速的目的。
蒸馏（distilling）将大模型中的知识迁移到小模型中，使小模型更易于训练。与上述方法相比，本文主要聚焦于设计更好的模型来提高性能，
而不是加速或迁移已有的模型。 <br>

### 3. Approach
#### 3.1 Channel Shuffle for Group Convolutions
Xception和ResNeXt分别引进了深度可分离卷积和分组卷积来权衡模型表示能力与计算量。但是这些设计都没有考虑其中的1x1卷积（也被称为逐点卷积(pointwise 
convolutions)），这部分也是需要很大的计算量的。举例来说，ResNeXt中只有3x3卷积采用分组卷积，那么每个残差block中93.4%的乘加计算来自于逐点卷积，
在极小的网络中逐点卷积会限制通道的数量，进而影响到模型性能。 <br>
为了解决这个问题，一个简单的解决方法就是在通道维度上应用稀疏连接，比方说在1x1卷积上也采用分组卷积的方式。但是这样做会带了副作用：
输出中的每个通道数据只是由输入中很小一部分的通道数据推导得到的（如图1(a)所示），这会阻碍信息在不同分组的通道间的流动，减弱网络的表示能力。 <br>
![](./data/figure_1.png)
如果让每个组的卷积可以获得其他组的输入数据（如图1(b)所示），那么输入/输出的各通道就是完全相关的。































