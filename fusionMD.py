import torch
import torch.nn as nn
import torch.nn.functional as F

class Feature_Pool(nn.Module):
    def __init__(self, dim, ratio=2):
        super(Feature_Pool, self).__init__()
        self.gap_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Linear(dim, dim * ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim * ratio, dim)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.gap_pool(x).view(b, c)
        y = self.up(self.act(self.down(y)))
        return y.view(b, c, 1, 1)


class Channel_Attention(nn.Module):
    def __init__(self, dim, ratio=16):
        super(Channel_Attention, self).__init__()
        self.gmp_pool = nn.AdaptiveMaxPool2d(1)
        self.down = nn.Linear(dim, dim // ratio)
        self.act = nn.GELU()
        self.up = nn.Linear(dim // ratio, dim)

    def forward(self, x):
        max_out = self.up(self.act(self.down(self.gmp_pool(x).permute(0, 2, 3, 1)))).permute(0, 3, 1, 2)
        return max_out


class Spatial_Attention(nn.Module):
    def __init__(self, dim):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x):
        x1 = self.conv1(x)
        return x1

class DualPathFusion(nn.Module):
    """融合超图XLSTM特征和频域特征"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        self.hyper_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.freq_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.fusion_gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )

    def forward(self, hyper_feat, freq_feat):

        hyper_enh = self.hyper_enhance(hyper_feat)
        freq_enh = self.freq_enhance(freq_feat)

        cat_feat = torch.cat([hyper_enh, freq_enh], dim=1)
        gate = self.fusion_gate(cat_feat)

        fused_feat = gate * hyper_enh + (1 - gate) * freq_enh
        out = self.final_conv(torch.cat([fused_feat, hyper_enh + freq_enh], dim=1))
        return out

class EAEF_new(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.mlp_pool = Feature_Pool(dim)
        self.dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=7, padding=3, groups=dim)
        self.T_c = nn.Parameter(torch.ones([]) * dim)
        self.cse = Channel_Attention(dim * 2)
        self.dual_fusion = DualPathFusion(dim)

    def forward(self, T, RGB):
        b, c, h, w = RGB.size()
        rgb_y = self.mlp_pool(RGB)
        t_y = self.mlp_pool(T)
        rgb_y = rgb_y / rgb_y.norm(dim=1, keepdim=True)
        t_y = t_y / t_y.norm(dim=1, keepdim=True)
        rgb_y = rgb_y.view(b, c, 1)
        t_y = t_y.view(b, 1, c)
        logits_per = self.T_c * rgb_y @ t_y
        cross_gate = torch.diagonal(torch.sigmoid(logits_per)).reshape(b, c, 1, 1)
        add_gate = torch.ones_like(cross_gate) - cross_gate

        New_RGB_A = RGB * cross_gate
        New_T_A = T * cross_gate

        x_cat = torch.cat((New_RGB_A, New_T_A), dim=1)
        fuse_gate = torch.sigmoid(self.cse(self.dwconv(x_cat)))
        rgb_gate, t_gate = fuse_gate[:, 0:c, :], fuse_gate[:, c:c * 2, :]

        New_RGB = RGB * add_gate + New_RGB_A * rgb_gate
        New_T = T * add_gate + New_T_A * t_gate

        out = self.dual_fusion(New_RGB, New_T)
        return out