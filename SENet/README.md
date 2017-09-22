# SENet
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507) <br>
Jie Hu, Li Shen, Gang Sun <br>

### 摘要
卷积神经网络顾名思义就是依赖卷积操作，使用局部感受区域（local receptive field）的思想融合空间信息和通道信息来提取包含信息的特征。
有很多工作从增强空间维度编码的角度来提升网络的表示能力，本文主要聚焦于通道维度，并提出一种新的结构单元——“Squeeze-and-Excitation(SE)”单元，
对通道间的依赖关系进行建模，可以自适应的调整各通道的特征响应值。如果将SE block添加到之前的先进网络中，只会增加很小的计算消耗，
但却可以极大地提升网络性能。依靠SENet作者获得了ILSVRC2017分类任务的第一名，top-5错误率为2.251%。 <br>

### 1. Introduction






























