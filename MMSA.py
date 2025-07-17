import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .prepbn import RepBN, LinearNorm
from ..modules.transformer import TransformerEncoderLayer, AIFI
from ..modules.conv import Conv, autopad
from einops import rearrange


__all__ = ['AIFI_RepBN', 'MMSA']

ln = nn.LayerNorm
linearnorm = partial(LinearNorm, norm1=ln, norm2=RepBN, step=60000)

class TransformerEncoderLayer_RepBN(TransformerEncoderLayer):
    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=..., normalize_before=False):
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)
        
        self.norm1 = linearnorm(c1)
        self.norm2 = linearnorm(c1)

class AIFI_RepBN(TransformerEncoderLayer_RepBN):
    """Defines the AIFI transformer layer."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0, act=nn.GELU(), normalize_before=False):
        """Initialize the AIFI instance with specified parameters."""
        super().__init__(c1, cm, num_heads, dropout, act, normalize_before)

    def forward(self, x):
        """Forward pass for the AIFI transformer layer."""
        c, h, w = x.shape[1:]
        pos_embed = self.build_2d_sincos_position_embedding(w, h, c)
        # Flatten [B, C, H, W] to [B, HxW, C]
        x = super().forward(x.flatten(2).permute(0, 2, 1), pos=pos_embed.to(device=x.device, dtype=x.dtype))
        return x.permute(0, 2, 1).view([-1, c, h, w]).contiguous()

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.0):
        """Builds 2D sine-cosine position embedding."""
        assert embed_dim % 4 == 0, "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="ij")
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], 1)[None]


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

######################################## TransformerEncoderLayer_AdditiveTokenMixer start ########################################

class MutilScal(nn.Module):
    def __init__(self, dim=512, fc_ratio=4, dilation=[3, 5, 7], pool_ratio=16):
        super(MutilScal, self).__init__()
        self.conv0_1 = Conv(dim, dim//fc_ratio)
        self.conv0_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-3], g=dim//fc_ratio)
        self.conv0_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv1_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-2], g=dim // fc_ratio)
        self.conv1_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv2_2 = Conv(dim//fc_ratio, dim//fc_ratio, 3, d=dilation[-1], g=dim//fc_ratio)
        self.conv2_3 = Conv(dim//fc_ratio, dim, 1)

        self.conv3 = Conv(dim, dim, 1)

        self.Avg = nn.AdaptiveAvgPool2d(pool_ratio)

    def forward(self, x):
        u = x.clone()

        attn0_1 = self.conv0_1(x)
        attn0_2 = self.conv0_2(attn0_1)
        attn0_3 = self.conv0_3(attn0_2)

        attn1_2 = self.conv1_2(attn0_1)
        attn1_3 = self.conv1_3(attn1_2)

        attn2_2 = self.conv2_2(attn0_1)
        attn2_3 = self.conv2_3(attn2_2)

        attn = attn0_3 + attn1_3 + attn2_3
        attn = self.conv3(attn)
        attn = attn * u

        pool = self.Avg(attn)

        return pool

class Mutilscal_MHSA(nn.Module):
    def __init__(self, dim, num_heads=2, atten_drop = 0., proj_drop = 0., dilation = [3, 5, 7], fc_ratio=4, pool_ratio=16):
        super(Mutilscal_MHSA, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.atten_drop = nn.Dropout(atten_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.MSC = MutilScal(dim=dim, fc_ratio=fc_ratio, dilation=dilation, pool_ratio=pool_ratio)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim//fc_ratio, kernel_size=1),
            nn.ReLU6(),
            nn.Conv2d(in_channels=dim//fc_ratio, out_channels=dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.kv = Conv(dim, 2 * dim, 1)

    def forward(self, x):
        u = x.clone()
        B, C, H, W = x.shape
        kv = self.MSC(x)
        kv = self.kv(kv)

        B1, C1, H1, W1 = kv.shape

        q = rearrange(x, 'b (h d) (hh) (ww) -> (b) h (hh ww) d', h=self.num_heads,
                      d=C // self.num_heads, hh=H, ww=W)
        k, v = rearrange(kv, 'b (kv h d) (hh) (ww) -> kv (b) h (hh ww) d', h=self.num_heads,
                         d=C // self.num_heads, hh=H1, ww=W1, kv=2)

        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.atten_drop(attn)
        attn = attn @ v

        attn = rearrange(attn, '(b) h (hh ww) d -> b (h d) (hh) (ww)', h=self.num_heads,
                         d=C // self.num_heads, hh=H, ww=W)
        c_attn = self.avgpool(x)
        c_attn = self.fc(c_attn)
        c_attn = c_attn * u
        return attn + c_attn

class MMSA(nn.Module):
    """Defines a single layer of the transformer encoder."""

    def __init__(self, c1, cm=2048, num_heads=8, dropout=0.0, act=nn.GELU(), normalize_before=False):
        """Initialize the TransformerEncoderLayer with specified parameters."""
        super().__init__()
        self.additivetoken = Mutilscal_MHSA(c1)
        # Implementation of Feedforward model
        self.fc1 = nn.Conv2d(c1, cm, 1)
        self.fc2 = nn.Conv2d(cm, c1, 1)

        self.norm1 = LayerNorm(c1)
        self.norm2 = LayerNorm(c1)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.act = act
        self.normalize_before = normalize_before

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Performs forward pass with post-normalization."""
        src2 = self.additivetoken(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.fc2(self.dropout(self.act(self.fc1(src))))
        src = src + self.dropout2(src2)
        return self.norm2(src)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        """Forward propagates the input through the encoder module."""
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

######################################## TransformerEncoderLayer_AdditiveTokenMixer end ########################################
