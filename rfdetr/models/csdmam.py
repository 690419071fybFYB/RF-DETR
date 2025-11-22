import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from torch.cuda.amp import autocast
class FCSA(nn.Module):
    """
    Frequency–Channel–Spatial Attention
    对应论文中的 FCSA 分支：
    - 频域增强：FFT + 频域通道权重
    - 空间-通道注意力：CA + SA
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 1)

        # 频域通道权重：GAP -> 1x1 Conv -> ReLU -> 1x1 Conv -> Sigmoid
        self.freq_ca = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空间-通道注意力：在 IFFT 之后
        self.channel_ca = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力：先做通道平均，再 1x1 卷积
        self.spatial_sa = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        with autocast(enabled=False):
            orig_dtype = x.dtype
            x32 = x.to(torch.float32)
            b, c, h, w = x32.size()

            # ===== 1. 频域分支：FFT → 频域通道增强 =====
            # 2D FFT（只对空间维度做）
            x_fft = torch.fft.rfft2(x32, norm="ortho")          # [B, C, H, W/2+1] 复数
            mag = torch.abs(x_fft)                            # 幅度用于求权重

            # GAP + 通道注意力（频域）
            freq_desc = self.gap(mag)                         # [B, C, 1, 1]
            freq_w = self.freq_ca(freq_desc)                  # [B, C, 1, 1]，Sigmoid 后 ∈ (0,1)

            # 复数乘法：实数权重直接与复数相乘
            x_fft_enh = x_fft * freq_w

            # IFFT 回到空间域
            x_ifft = torch.fft.irfft2(x_fft_enh, s=(h, w), norm="ortho")  # [B, C, H, W] 实数
            x_ifft = x_ifft.to(orig_dtype)

        # ===== 2. 空间 + 通道注意力 =====
        # 通道注意力（空间域）
        ch_desc = self.gap(x_ifft)                        # [B, C, 1, 1]
        ch_w = self.channel_ca(ch_desc)                   # [B, C, 1, 1]

        # 空间注意力
        spa_map = x_ifft.mean(dim=1, keepdim=True)        # [B, 1, H, W]
        spa_w = self.spatial_sa(spa_map)                  # [B, 1, H, W]

        out = x_ifft * ch_w * spa_w                       # 同时做 C 和 H,W reweight
        return out
    
class DepthwiseConv(nn.Module):
    """简单的深度可分离卷积封装"""
    def __init__(self, channels, kernel_size, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.dw = nn.Conv2d(
            channels, channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=False
        )

    def forward(self, x):
        return self.dw(x)


class DMAM(nn.Module):
    """
    Dual-Domain Multi-Scale Attention Module
    对应图中 (a) 的 DMAM：
    - 输入先 Conv 1x1
    - Local 分支：depthwise conv (小核)
    - Large 分支：三个方向的大核 depthwise conv (31x1, 31x31, 1x31)
    - Global 分支：FCSA
    - 分支求和后再 Conv 1x1
    """
    def __init__(self, channels, k_large=31, reduction=8):
        super().__init__()
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Local：这里按图里写成 DConv 1x1（你也可以改成 3x3）
        self.local = DepthwiseConv(channels, kernel_size=1, padding=0)

        # Large branch：三种方向的大核 depthwise conv
        self.dconv_31x1 = DepthwiseConv(
            channels,
            kernel_size=(k_large, 1),
            padding=(k_large // 2, 0)
        )
        self.dconv_1x31 = DepthwiseConv(
            channels,
            kernel_size=(1, k_large),
            padding=(0, k_large // 2)
        )
        self.dconv_31x31 = DepthwiseConv(
            channels,
            kernel_size=(k_large, k_large),
            padding=k_large // 2
        )

        # Global：FCSA
        self.fcsa = FCSA(channels, reduction=reduction)

        # 输出融合 1x1 卷积
        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU(inplace=True)  # 你也可以换成 ReLU / GELU

    def forward(self, x):
        x = self.conv_in(x)

        # Local
        x_local = self.local(x)

        # Large (三种大核方向的和)
        x_l1 = self.dconv_31x1(x)
        x_l2 = self.dconv_1x31(x)
        x_l3 = self.dconv_31x31(x)
        x_large = x_l1 + x_l2 + x_l3

        # Global (FCSA)
        x_global = self.fcsa(x)

        # 多分支融合
        out = x_local + x_large + x_global
        out = self.conv_out(out)
        out = self.bn(out)
        out = self.act(out)
        return out
    
class CSDMAM(nn.Module):
    """
    Cross-Stage Dual-Domain Multi-Scale Attention Module
    对应图中 (c)：
    x ----> + ------------------> out
      \     ^
       Conv1x1 -> DMAM ---------/
    """
    def __init__(self, channels, k_large=31, reduction=8):
        super().__init__()
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.dmam = DMAM(channels, k_large=k_large, reduction=reduction)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.pre_conv(x)
        out = self.dmam(out)

        out = out + identity          # 残差 Add
        out = self.bn(out)
        return out
    
if __name__ == "__main__":
    from torchinfo import summary
    csdmam = CSDMAM(256)
    summary(csdmam, input_size=(1, 256, 80, 80),depth=3)
