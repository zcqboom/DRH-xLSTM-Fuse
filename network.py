from utils.cuda_chk import GPUorCPU
DEVICE = GPUorCPU.DEVICE
from XLSTM import *
from r1 import *
from fusionMD import *

class SimAM(nn.Module):

    def __init__(self, lamda=1e-5):
        super().__init__()
        self.lamda = lamda
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.shape
        n = h * w - 1
        mean = torch.mean(x, dim=[-2, -1], keepdim=True)
        var = torch.sum(torch.pow((x - mean), 2), dim=[-2, -1], keepdim=True) / n
        e_t = torch.pow((x - mean), 2) / (4 * (var + self.lamda)) + 0.5
        out = self.sigmoid(e_t) * x
        return out


class SimAMFusion(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.simam = SimAM()

    def forward(self, A, B):
        x = torch.cat([A, B], dim=1)
        x = self.conv(x)
        x = self.simam(x)
        return x

class MyNetwork(nn.Module):
    def __init__(self, img_channels=3, dropout=0.):
        super().__init__()
        self.conv = nn.Conv2d(img_channels, 32, kernel_size=3, padding=1, stride=1)
        
        config = HGMambaConfig(
            d_model=32,
            top_k=9,
            n_sequence=8,
            arw_length_ratio=0.7,
            n_layers=2,
            tile_size=16,
            ssm_d_state=16,
            ssm_d_conv=4,
            ssm_expand=2
        )
        self.hgnn = HGXLSMFeatureExtractor(config=config, in_channels=3).to(DEVICE)

        self.mix_fusion1 = SimAMFusion(channels=32)
        self.mix_fusion2 = SimAMFusion(channels=32)

        self.frq1 = FrequencyDecompositionAdaptiveFusionModule(dim=32)

        self.frq2 = FrequencyDecompositionAdaptiveFusionModule(dim=32)

        self.frequency1 = FrequencyRefinementModule(dim=32)
        self.frequency2 = FrequencyRefinementModule(dim=32)

        self.frRe = AdaptiveFrequencyReconstruction(dim=32, wave='haar')

        self.Decoder = EAEF_new(32)
        self.Reconstruct = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1, stride=1),
            nn.Sigmoid())



    def forward(self, A, B):
        _, _, H, W = A.shape
        pad_h = (H % 2)
        pad_w = (W % 2)
        if pad_h or pad_w:
            A = nn.functional.pad(A, (0, pad_w, 0, pad_h), mode='replicate')
            B = nn.functional.pad(B, (0, pad_w, 0, pad_h), mode='replicate')
        (A_G, A_fea), (B_G, B_fea) = self.hgnn(A, B)
        F_G = self.mix_fusion2(A_G, B_G)
        A = self.conv(A)
        B = self.conv(B)
        L_lf, L_hf = self.frq1(A, B)
        H_lf, H_hf = self.frq1(A_fea[0], B_fea[0])
        L_lf = L_lf + H_lf
        L_hf = L_hf + H_hf
        lf_feat, hf_feat = self.frequency1(L_lf, L_hf)
        H_lf, H_hf = self.frq2(A_fea[1], B_fea[1])
        lf_feat = lf_feat + H_lf
        hf_feat = hf_feat + H_hf
        lf_feat, hf_feat = self.frequency2(lf_feat, hf_feat)
        F_freq = self.frRe(lf_feat, hf_feat)
        F = self.Decoder(F_freq, F_G)
        F = self.Reconstruct(F)
        if pad_h or pad_w:
            F = F[:, :, :H, :W]
        return F


if __name__ == '__main__':
    test_tensor_A = torch.rand((1, 3, 224, 224)).to(DEVICE)
    test_tensor_B = torch.rand((1, 3, 224, 224)).to(DEVICE)
    model = MyNetwork().to(DEVICE)
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print(model)
    print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
    NetOut = model(test_tensor_A, test_tensor_B)
    print(NetOut.shape)
    