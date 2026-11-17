import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from torchvision.utils import save_image
from dwt_function import DWT_2D, IDWT_2D

class FocusDegreeCalculator(nn.Module):

    def __init__(self, kernel_size=3):
        super().__init__()
        self.sobel_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.sobel_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_x.weight = nn.Parameter(sobel_x_kernel, requires_grad=False)
        self.sobel_y.weight = nn.Parameter(sobel_y_kernel, requires_grad=False)

    def forward(self, x):
        gray = torch.mean(x, dim=1, keepdim=True)
        grad_x = self.sobel_x(gray)
        grad_y = self.sobel_y(gray)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.max() - grad_mag.min() + 1e-6)
        return grad_mag

class FrequencyDecompositionAdaptiveFusionModule(nn.Module):
    def __init__(self, dim, wave='haar'):
        super().__init__()
        self.dwt = DWT_2D(wave=wave)
        self.focus_calc = FocusDegreeCalculator()

        self.lf_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

        self.hf_gate = nn.Sequential(
            nn.Conv2d(1, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, vi_features, ir_features):
        import torch.nn.functional as F
        dwt_vi = self.dwt(vi_features)
        dwt_ir = self.dwt(ir_features)

        B, C4, Hf, Wf = dwt_vi.shape
        C = C4 // 4
        LF_vi, HF_vi = dwt_vi[:, :C, :, :], dwt_vi[:, C:, :, :]
        LF_ir, HF_ir = dwt_ir[:, :C, :, :], dwt_ir[:, C:, :, :]

        focus_vi = self.focus_calc(vi_features)
        focus_ir = self.focus_calc(ir_features)

        focus_vi = F.interpolate(focus_vi, size=(Hf, Wf), mode='bilinear', align_corners=False)
        focus_ir = F.interpolate(focus_ir, size=(Hf, Wf), mode='bilinear', align_corners=False)

        lf_cat = torch.cat([LF_vi, LF_ir], dim=1)
        lf_weight = self.lf_gate(lf_cat)
        LF_fuse = lf_weight * LF_vi + (1 - lf_weight) * LF_ir

        hf_weight_vi = self.hf_gate(focus_vi)
        hf_weight_ir = self.hf_gate(focus_ir)
        hf_weight = hf_weight_vi / (hf_weight_vi + hf_weight_ir + 1e-6)

        HF_fuse_list = []
        for i in range(3):
            hf_vi = HF_vi[:, i * C:(i + 1) * C, :, :]
            hf_ir = HF_ir[:, i * C:(i + 1) * C, :, :]
            hf_fuse = hf_weight * hf_vi + (1 - hf_weight) * hf_ir
            HF_fuse_list.append(hf_fuse)
        HF_fuse = torch.cat(HF_fuse_list, dim=1)

        return LF_fuse, HF_fuse


class FrequencySpatialInteractionAttention(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.dim = dim
        self.spatial_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.spatial_norm = nn.LayerNorm(dim)
        self.freq_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // reduction, 1),
            nn.GELU(),
            nn.Conv2d(dim // reduction, dim, 1),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        B, C, H, W = x.shape
        x_spatial = self.spatial_conv(x)
        x_spatial = x_spatial.permute(0, 2, 3, 1)
        x_spatial = self.spatial_norm(x_spatial)
        x_spatial = x_spatial.permute(0, 3, 1, 2)

        x_freq = self.freq_proj(x)

        alpha = torch.sigmoid(self.alpha)
        out = alpha * x_spatial * x_freq + (1 - alpha) * x
        return out

class AdaptiveFrequencyReconstruction(nn.Module):
    def __init__(self, dim, wave='haar', groups=4):
        super().__init__()
        self.dim = dim
        self.groups = groups
        self.idwt = IDWT_2D(wave=wave)
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, lf_feat, hf_feat, guide_feat=None):
        dwt_tensor = torch.cat([lf_feat, hf_feat], dim=1)
        base_feat = self.idwt(dwt_tensor)
        edge_feat = self.edge_enhance(base_feat)
        final_feat = self.fusion_conv(base_feat + edge_feat)
        return final_feat


class LowFrequencyFusionBlock(nn.Module):
    """原有模块保持不变"""

    def __init__(self, dim):
        super().__init__()
        self.conv_before = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 7), padding=(0, 3), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(7, 1), padding=(3, 0), groups=dim),
            nn.SiLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=(1, 11), padding=(0, 5), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(11, 1), padding=(5, 0), groups=dim),
            nn.SiLU()
        )
        self.ms_fuse = nn.Conv2d(dim, dim, 1)
        self.pool_h_avg = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w_avg = nn.AdaptiveAvgPool2d((1, None))
        self.pool_h_max = nn.AdaptiveMaxPool2d((None, 1))
        self.pool_w_max = nn.AdaptiveMaxPool2d((1, None))
        self.conv_h = nn.Sequential(nn.Conv2d(dim * 2, dim, (3, 1), padding=(1, 0)), nn.SiLU())
        self.conv_w = nn.Sequential(nn.Conv2d(dim * 2, dim, (1, 3), padding=(0, 1)), nn.SiLU())
        self.attn_fusion = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Sigmoid()
        )
        self.conv_after = nn.Conv2d(dim, dim, 1)
        self.res_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, LF_in):
        main_residual = LF_in
        x = self.conv_before(LF_in)
        feat_ms = self.ms_fuse(self.branch1(x) + self.branch2(x))
        h_stat = torch.cat([self.pool_h_avg(x), self.pool_h_max(x)], dim=1)
        w_stat = torch.cat([self.pool_w_avg(x), self.pool_w_max(x)], dim=1)
        attn_map = self.attn_fusion(self.conv_h(h_stat) * self.conv_w(w_stat))
        internal_residual = x + feat_ms * attn_map
        out = self.conv_after(internal_residual)
        return main_residual + self.res_scale * out


class GuidedFilterHFBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        from torchvision.transforms import GaussianBlur
        self.gaussian_3 = GaussianBlur(kernel_size=3, sigma=0.5)
        self.gaussian_5 = GaussianBlur(kernel_size=5, sigma=1.0)
        self.conv_before = nn.Sequential(nn.Conv2d(3 * dim, 3 * dim, 1), nn.SiLU())
        self.conv_edge = nn.Sequential(nn.Conv2d(3 * dim, 3 * dim, 3, padding=1), nn.SiLU())
        self.gate = nn.Sequential(nn.Conv2d(3 * dim, 3 * dim, 1), nn.Sigmoid())
        self.conv_out = nn.Sequential(nn.Conv2d(3 * dim, 3 * dim, 1), nn.SiLU())

    def forward(self, x):
        x = self.conv_before(x)
        blur3 = self.gaussian_3(x)
        blur5 = self.gaussian_5(x)
        edge3 = x - blur3
        edge5 = x - blur5
        edge = edge3 + edge5
        edge = self.conv_edge(edge)
        gate = self.gate(x)
        out = x + gate * edge
        out = self.conv_out(out)
        return out


class FrequencyRefinementModule(nn.Module):

    def __init__(self, dim, wave='haar'):
        super().__init__()
        self.LFB = LowFrequencyFusionBlock(dim=dim)
        self.GFHB = GuidedFilterHFBlock(dim=dim)

    def forward(self, LF_fuse, HF_fuse):
        fused_LF = self.LFB(LF_fuse)
        fused_HF = self.GFHB(HF_fuse)
        return fused_LF, fused_HF