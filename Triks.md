# Tricks

## 一、分类

### 1、长尾问题

平衡二元交叉熵是一种用于处理不平衡数据的损失函数。在二元分类问题中，如果数据集中正负样本的数量差异很大，那么标准的二元交叉熵损失函数可能会导致模型偏向于学习少数类别的特征，而忽略多数类别的特征。为了解决这个问题，平衡二元交叉熵损失函数通过对 Sigmoid 函数进行无偏扩展，并给每个类别分配一个额外的对数边际，来减少少数类别和多数类别之间的差距，从而提高网络的分类性能。
具体来说，平衡二元交叉熵损失函数可以表示为：
$$
\begin{aligned}
\text{Balanced Binary Cross-Entropy Loss} &= -\frac{1}{N}\sum_{i=1}^{N}\left[\alpha_t y_i \log(p(y_i)) + (1-\alpha_t)(1-y_i)\log(1-p(y_i))\right] \\
\alpha_t &= \frac{\sum_{i=1}^{N}(1-y_i)}{\sum_{i=1}^{N}y_i}
\end{aligned}
$$
其中 $y_i$ 是第 $i$ 个样本的真实标签，$p(y_i)$ 是模型对第 $i$ 个样本属于正类的预测概率，$\alpha_t$ 是一个动态调整的权重系数，用于平衡正负样本之间的数量差异。$\alpha_t$ 的计算公式为 $\alpha_t = \frac{\sum_{i=1}^{N}(1-y_i)}{\sum_{i=1}^{N}y_i}$，其中 $\sum_{i=1}^{N}(1-y_i)$ 表示负样本的数量，$\sum_{i=1}^{N}y_i$ 表示正样本的数量。
相比于标准的二元交叉熵损失函数，平衡二元交叉熵损失函数可以更好地适应长尾数据分布并学习到更通用的特征表示。

## 二、训练

### 1、大训练批不稳定

A Simple Framework for Contrastive Learning of Visual Representations

> Training with large batch size may be unstable when using standard SGD/Momentum with linear learning rate scaling (Goyal et al., 2017). To stabilize the training, we use the LARS optimizer (You et al., 2017) for all batch sizes.
