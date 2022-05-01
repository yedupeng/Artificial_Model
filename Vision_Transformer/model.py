from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

def drop_path(x, drop_prob:float = 0, training: float = 0):                                   # 用于随机深度
    if drop_prob==0 or not training:
        return x
    keep_prob = 1-drop_prob
    shape = (x.shape[0],)+(1,)*(x.ndim-1)
    random_tensor = keep_prob-torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob)*random_tensor
    return output

class Drop_path(nn.Module):
    def __init__(self, drop_prob=None):
        super(Drop_path, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Attention(nn.Module):                                                                  # 编写attention操作用于encoding层中的multi_head层
    def __init__(self,
                 dim,                                                                        # 输入的dimension
                 num_head=8,                                                                 # multi_head中head的个数
                 qkv_bias=False,
                 qkv_scale=None,
                 attn_drop_ratio=0.,
                 prob_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_head = num_head
        head_dim = dim//num_head                                                             # 计算每个head中的dimension
        self.scale = qkv_scale or head_dim**-0.5                                             # attention计算式中的根号d
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)                                      # 全连接层得到qkv总体
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)                                                      # 对每个head拼接以后的结果进行映射
        self.proj_drop = nn.Dropout(prob_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_head, C//self.num_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                                                     # 分别得到q、k、v
        attn = (q @ k.transpose(-2, -1)) * self.scale                                        # 根据式子进行计算
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)                                      # 利用reshape进行拼接
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):                                                                       # encoding层中的MLP层
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        hidden_features = hidden_features or in_features                                    # 注意MLP层中，第一个全连接层的输出是输入的4倍
        out_features = out_features or in_features
        self.f1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.f2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.f1(x)
        x = self.act(x)
        x = self.f2(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class PathEmbed(nn.Module):                                                                  # 第一层   pathembedding层
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None): # 输入图像大小  卷积核大小  输入通道数  卷积核个数  归一化参数
        super().__init__()
        img_size = (img_size, img_size)                                                      # 224*224
        patch_size = (patch_size, patch_size)                                                # 16*16
        self.img_size = img_size
        self.grid_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])            # (224//16,224//16)->(14,14)
        self.num_patch = self.grid_size[0]*self.grid_size[1]
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()                   # 如果有初始化含数则调用，若无则不做操作

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
                     f"Imput image size ({H}*{W}) does not match model({self.img_size[0],self.img_size[1]})."      #判断是否图像大小调整操作有问题
        x = self.proj(x).flatten(2).transpose(1, 2)                                          # flatten：[B, C, H, W]->[B, C, HW]->transpose->[B, HW, C]
        x = self.norm(x)
        return x



class Block(nn.Module):                                                                      # 第二层 encoding层
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,                                                              # MLP层第一层全连接层输出对于输入的倍率
                 qkv_bias=False,
                 qkv_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)                                                          # 第一层Layer_Norm层
        self.attn = Attention(dim, num_head=num_heads, qkv_bias=qkv_bias, qkv_scale=qkv_scale,# 第一层Multi_Head_Attention层
                              attn_drop_ratio=attn_drop_ratio, prob_drop_ratio=drop_ratio)    # 第一层Dropout层
        self.drop_path = Drop_path(drop_path_ratio) if drop_path_ratio>0. else nn.Identity()
        self.norm2 = norm_layer(dim)                                                          # 第二层Layer_Norm层
        mlp_hidden_dim = int(dim * mlp_ratio)                                                 # Mlp层
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):                                                                     # 按照网络的残差形式输出
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class VisitionTransformer(nn.Module):                                                         # 编写网络结构
    def __init__(self,
                 img_size=16, patch_size=16, int_c=3, num_class=1000,                         # 给定各层参数
                 embed_dim=768, depth=12, num_head=12, mlp_ratio=4.0,
                 qkv_bias=True, qk_scale=None, representation=None,
                 distilled=False, drop_ratio=0, attn_drop_ratio=0.,
                 drop_path_ratio=0., embed_lay=PathEmbed, norm_layer=None, act_layer=None):
        super(VisitionTransformer, self).__init__()
        self.num_class = num_class
        self.num_feature = self.embed_dim = embed_dim
        self.num_take = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)                            # 传入参数1e-6
        act_layer = act_layer or nn.GELU
        self.patch_embed = embed_lay(img_size=img_size, patch_size=patch_size,                # 第一层   pathembedding层
                                     in_c=int_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patch
        self.cla_taken = nn.Parameter(torch.zeros(1, 1, embed_dim))                           # 结构中的class_taken
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None   # 此行默认为None 不起作用
        self.pose_embed = nn.Parameter(torch.zeros((1, num_patches+self.num_take, embed_dim)))# 加入位置编码
        self.pose_drop = nn.Dropout(p=drop_ratio)                                             # 上述操作后所要进行的dropout操作

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        self.block = nn.Sequential(*[                                                         # 第二层 encoding层
            Block(dim=embed_dim, num_heads=num_head, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qkv_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i], norm_layer=norm_layer,
                  act_layer=act_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)                                                     # 第二层结束后的Layer_Norm操作
        if representation and not distilled:                                                  # 判断第三层MLP_head中是否有Pre_logits
            self.has_logits = True                                                            # 若有Pre_logits则编写结构，若无则nn.Identity（）不进行任何操作
            self.num_feature = representation
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_feature, num_class) if num_class > 0 else nn.Identity()# 第三层MLP_head中的全连接层
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_class) if num_class > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pose_embed, std=0.02)                                      # 权重初始化
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cla_taken, std=0.02)
        self.apply(_init_vit_weights)

    def _forward(self, x):
        x = self.patch_embed(x)                                                              # 第一层   pathembedding层
        cls_token = self.cla_taken.expand(x.shape[0], -1, -1)
        if self.dist_token is None:                                                          # 拼接class_taken
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pose_drop(x + self.pose_embed)                                              # 加入位置编码
        x = self.block(x)                                                                    # 第二层 encoding层
        x = self.norm(x)                                                                     # 第二层结束后的Layer_Norm操作
        if self.dist_token is None:
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):                                                                    # 完成前两层后第三层输出
        x = self._forward(x)
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.traning and not torch.jit.is_scripting():
                return x, x_dist
            else:
                return (x+x_dist)//2
        else:
            x = self.head(x)
        return x

def vit_base_patch16_224_in21k(num_class: int=21843, has_logits: bool=True):
    model = VisitionTransformer(img_size=224,
                                patch_size=32,
                                embed_dim=768,
                                depth=12,
                                num_head=12,
                                representation=768 if has_logits else None,
                                num_class=num_class)
    return model

def _init_vit_weights(m):                                                                    # 初始化权重函数
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)



