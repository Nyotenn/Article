# ViT

## 一、Introduction

> 在计算机视觉领域，`Attention` 机制要么和卷积网络一起使用，要么只是替代卷积网络的部分并保持总体结构不变。
>
> `ViT` 的提出，表明对卷积网络的依赖是不必要的，可以直接将图片打成数个 `patches`，将一个纯粹的 `Transformer` 应用于这些 `patches` 之上，仍然能在图像分类任务上有良好表现。

## 二、Related Works

- `Image Transformer` 仅在邻域而非全局上采用 self-attention ，这样的多头注意力机制取代了卷积操作。
- `Sparse Transformer` 选择稀疏 Transformer 模型中的 attention 矩阵。

## 三、Coding

- **image_size**：图片尺寸，int或者tuple皆可，例如224，或者(224, 224)，长宽不一定要一样大
- **path_size**：*分块path尺寸，int或tuple，默认为16，需要确保*image_size*能被*path_size*整除
- **num_classes**：分类数，int
- **dim**：Transformer隐层维度，对于Base来说是768，Large为1024
- **depth**：Transformer个数，Base为12
- **head**：多头的个数，Base=12
- **mlp_dim**：Transformer中的FeedForward中第一个线性层升维后的维度，默认为768*4，先升维4倍再降维回去
- **pool**：默认'cls，选取CLS token作为输出，可选'mean'，在patch维度做平均池化
- **channel**：图片输入的特征维度，RGB图像为3，灰度图为1

```py
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
      	  super().__init__()
        image_height, image_width = pair(image_size)  # 在这个项目中没有限定图片的尺寸
        patch_height, patch_width = pair(patch_size)  # 默认为16

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        # num patches -> (224 / 16) = 14, 14 * 14 = 196
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        # path dim -> 3 * 16 * 16 = 768，和Bert-base一致
        patch_dim = channels * patch_height * patch_width  
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'  # 输出选cls token还是做平均池化

        # 步骤一：图像分块与映射。首先将图片分块，然后接一个线性层做映射
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        # pos_embedding：位置编码；cls_token：在序列最前面插入一个cls token作为分类输出
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # 步骤二：Transformer Encoder结构来提特征
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()  # 这一																															步上面都没做

        # 线性层输出
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
    ...
```

