# GDA & Native Bayes

## 一、GDA

> GDA（Gaussian Discriminate Analysis，高斯判别分析）是分类算法的一种，其假设不同类别的样本均服从高斯分布，即
> $$
> x|y=0 \sim N(\mu_0, \Sigma)\\
> x|y=1 \sim N(\mu_0, \Sigma)
> $$
> 首先估计出先验概率以及多元高斯分布的均值和协方差矩阵，然后再由贝叶斯公式求出一个新样本分别属于两类别的概率，预测结果取概率值大者。

### 1、假设函数

假设有 m 个样本数据，样本数据满足：
$$
y \sim Bernoulli(\phi)\\
x|y=0 \sim N(\mu_0, \Sigma)\\
x|y=1 \sim N(\mu_0, \Sigma)
$$
即：
$$
p(y)=\phi^y(1-\phi)^{1-y}\\
p(x|y=0)=\frac1{(2\pi)^\frac n{2}|\Sigma|^\frac 1{n}}e^{-\frac1{2}(x-\mu_0)^T\Sigma^{-1}(x-\mu_0)}\\
p(x|y=1)=\frac1{(2\pi)^\frac n{2}|\Sigma|^\frac 1{n}}e^{-\frac1{2}(x-\mu_1)^T\Sigma^{-1}(x-\mu_1)}
$$
其中，未知参数为 $\phi、\Sigma、\mu_0、\mu_1$，

假设函数为 $P ( x ) P(y|x)=\dfrac{P(x|y)P(y)}{P(x)}$ ,由于对两种类别而言，概率大小仅取决于分子，故可将分母看作常量，分别计算 $P ( x ∣ y = 0 ) P ( y = 0 )$ 和 $P ( x ∣ y = 1 ) P ( y = 1 )$ 的概率，概率大者为样本数据所属类别。

### 2、损失函数

已知样本数据含有参数的概率分布，由最大似然估计法可以推导高斯判别分析模型的损失函数为$$ \begin{aligned} \mathcal L(\phi,\mu_0,\mu_1,\Sigma) &= \log\prod\limits_{i=1}^{m}p(x^{(i)},y^{(i)};\phi,\mu_0,\mu_1,\Sigma)\\ &=\log\prod\limits_{i=1}^{m}p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma)p(y^{(i)};\phi)\\ &=\sum\limits_{i=1}^{m}\log p(x^{(i)}|y^{(i)};\mu_0,\mu_1,\Sigma) + \sum\limits_{i=1}^{m}\log p(y^{(i)};\phi)\\ &=\sum\limits_{i=1}^{m}\log \left(p(x^{(i)}|y^{(i)}=1;\mu_1,\Sigma)^{y^{(i)}} \cdot p(x^{(i)}|y^{(i)}=0;\mu_0,\Sigma)^{1-y^{(i)}}\right)+ \sum\limits_{i=1}^{m}\log p(y^{(i)};\phi)\\ &=\sum\limits_{i=1}^{m}y^{(i)}\log p(x^{(i)}|y^{(i)}=1;\mu_1,\Sigma)+\sum\limits_{i=1}^{m}(1-y^{(i)})\log p(x^{(i)}|y^{(i)}=0;\mu_0,\Sigma) + \sum\limits_{i=1}^{m}\log p(y^{(i)};\phi)\ \end{aligned} $$

 对最大似然函数$\ \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)\ $求偏导：

$$\begin{aligned} \nabla_{\phi} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)&=\nabla_{\phi} \sum\limits_{i=1}^{m}\log p(y^{(i)};\phi)\\ &=\nabla_{\phi}\sum\limits_{i=1}^{m}\log\phi^{y^{(i)}}(1-\phi)^{(1-y^{(i)})}\\ &=\nabla_{\phi}\sum\limits_{i=1}^{m}{y^{(i)}\log\phi+(1-y^{(i)})\log(1-\phi)}\\ &=\sum\limits_{i=1}^{m}{y^{(i)}\cdot \dfrac{1}{\phi}-(1-y^{(i)})\cdot\dfrac{1}{1-\phi} }\\ &=\sum\limits_{i=1}^{m}{I(y^{(i)}=1)\cdot\dfrac{1}{\phi}-I(y^{(i)}=0)\cdot\dfrac{1}{1-\phi} } \end{aligned} $$ 

其中$\ I(x)\ $为示性函数，当$ \ x\ $为真时为真时$\ I(x)\ $的值为的值为$\ 1 \ $ 当$\ x\ $为假时为假时$\ I(x)\ $的值为的值为$\ 0\ $，令，令$\ \nabla_{\phi} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)=0\ $ $$ \begin{aligned} \phi=\dfrac{\sum\limits_{i=1}^{m}I(y^{(i)}=1)}{\sum\limits_{i=1}^{m}{I(y^{(i)}=1)+I(y^{(i)}=0)}}=\dfrac{\sum\limits_{i=1}^{m}I(y^{(i)}=1)}{m} \end{aligned} $$ 

同样地，对$\ \mu_0 \ $求偏导可得求偏导可得：

$$ \begin{aligned} \nabla_{\mu_0} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)&=\nabla_{\mu_0}\sum\limits_{i=1}^{m}(1-y^{(i)})\log p(x^{(i)}|y^{(i)}=0;\mu_0,\Sigma)\\ &=\nabla_{\mu_0}\sum\limits_{i=1}^{m}(1-y^{(i)})\cdot\log\dfrac{1}{(2\pi)^{\frac{n}{2}}\vert \Sigma \vert ^{\frac{1}{2}}}e^{-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0)}\\ &=\sum\limits_{i=1}^{m}(1-y^{(i)})\Sigma^{-1}(x^{(i)}-\mu_0)\\ &=\sum\limits_{i=1}^{m}(I(y^{(i)})=0)\Sigma^{-1}(x^{(i)}-\mu_0) \end{aligned} $$ 

令$\ \nabla_{\mu_0} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)=0\ $ $$ \begin{aligned} \mu_0=\dfrac{\sum\limits_{i=1}^{m}I(y^{(i)}=0)x^{(i)}}{\sum\limits_{i=1}^{m}I(y^{(i)}=0)} \end{aligned} $$ 

根据对称性可知 $$ \begin{aligned} \mu_1=\dfrac{\sum\limits_{i=1}^{m}I(y^{(i)}=1)x^{(i)}}{\sum\limits_{i=1}^{m}I(y^{(i)}=1)} \end{aligned} $$ 

最后对$\ \Sigma\ $求偏导可得：

$$ \begin{aligned} \nabla_{\Sigma} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)&=\nabla_{\Sigma} \left(\sum\limits_{i=1}^{m}y^{(i)}\log p(x^{(i)}|y^{(i)}=1;\mu_1,\Sigma)+\sum\limits_{i=1}^{m}(1-y^{(i)})\log p(x^{(i)}|y^{(i)}=0;\mu_0,\Sigma)\right)\\ &=\nabla_{\Sigma}\left( \sum\limits_{i=1}^{m}y^{(i)}\cdot\log\dfrac{1}{(2\pi)^{\frac{n}{2}}\vert \Sigma \vert ^{\frac{1}{2}}}e^{-\frac{1}{2}(x^{(i)}-\mu_1)^T\Sigma^{-1}(x^{(i)}-\mu_1)}+\ \sum\limits_{i=1}^{m}(1-y^{(i)})\cdot\log\dfrac{1}{(2\pi)^{\frac{n}{2}}\vert \Sigma \vert ^{\frac{1}{2}}}e^{-\frac{1}{2}(x^{(i)}-\mu_0)^T\Sigma^{-1}(x^{(i)}-\mu_0)}\right)\\ &=\nabla_{\Sigma}\left(\sum\limits_{i=1}^m\log\dfrac{1}{(2\pi)^{\frac{n}{2}}\vert \Sigma \vert ^{\frac{1}{2}}}-\dfrac{1}{2}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})\right)\\ &=\nabla_{\Sigma}\left(\sum\limits_{i=1}^m(-\dfrac{n}{2}\log2\pi-\dfrac{1}{2}\log\vert\Sigma\vert)-\dfrac{1}{2}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})^T\Sigma^{-1}(x^{(i)}-\mu_{y^{(i)}})\right)\\ &=-\dfrac{1}{2}\sum\limits_{i=1}^m\dfrac{1}{\vert\Sigma\vert}\vert\Sigma\vert\Sigma^{-1} -\dfrac{1}{2}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T\cdot\nabla_{\Sigma}\Sigma^{-1}\\ &=-\dfrac{m}{2}\Sigma^{-1}-\dfrac{1}{2}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T(-\Sigma^{-2}) \end{aligned} $$

其中直接利用了下面的结论 
$$
\begin{aligned} &\nabla_{\Sigma}\vert\Sigma\vert=\vert\Sigma\vert\Sigma^{-1}\\ 

&\nabla_{\Sigma}\Sigma^{-1}=-\Sigma^{-2} \end{aligned}
$$
令 $\nabla_{\Sigma} \mathcal{L}(\phi,\mu_0,\mu_1,\Sigma)=0$

则 $ \Sigma=\dfrac{1}{m}\sum\limits_{i=1}^m(x^{(i)}-\mu_{y^{(i)}})(x^{(i)}-\mu_{y^{(i)}})^T $

## 二、Native Bayes

> 朴素贝叶斯方法是在[贝叶斯](https://baike.baidu.com/item/贝叶斯/1405899)算法的基础上进行了相应的简化，即假定给定目标值时属性之间相互条件独立。

