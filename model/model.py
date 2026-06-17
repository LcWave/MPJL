import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def calc_rf(kernels, dilations, strides=None):
    """Compute the effective receptive field of stacked 2D Convs (simplified for stride=1)."""
    if strides is None:
        strides = [1]*len(kernels)
    rf = 1
    prod_stride = 1
    for k, d, s in zip(kernels, dilations, strides):
        rf += (k - 1) * d * prod_stride
        prod_stride *= s
    return rf

def period_to_conv_cfg_rfmatch(k, win_size, image_size,
                               depth=3, rf_ratio=1.5,
                               k_min_kernel=3, k_max_kernel=11,
                               d_min=1, d_max=8,
                               p_pix_override=None):
    """
    Receptive-field matching strategy:
      1) Compute the target RF
      2) Greedily pick (kernel, dilation) per layer until RF ≥ target
    """
    # Period measured in pixels
    if p_pix_override is None:
        P_pix = math.ceil(k / win_size * image_size)
    else:
        P_pix = int(p_pix_override)
    P_pix = max(1, P_pix)
    target_rf = math.ceil(P_pix * rf_ratio)

    strides = [1]*depth
    kernels = []
    dilations = []

    # Roughly split the remaining RF across the remaining layers
    remain = target_rf
    for i in range(depth):
        layers_left = depth - i
        # Desired per-layer contribution: (k-1)*d ≈ contrib
        # (k-1)*d ≈ contrib
        contrib = max(2, remain // layers_left)  # at least contribute 2
        # Search a feasible (k, d) within bounds
        best = (3, 1)  # fallback
        best_err = 1e9
        for ksize in range(k_min_kernel, k_max_kernel+1, 2):  # try odd kernels only
            for dil in range(d_min, d_max+1):
                this = (ksize - 1) * dil
                err = abs(this - contrib)
                if err < best_err:
                    best_err = err
                    best = (ksize, dil)
        kernels.append(best[0])
        dilations.append(best[1])

        # Update remain (subtract the rough contribution)
        remain = max(0, remain - (best[0]-1)*best[1])

    # If RF is still insufficient, enlarge the last layer's dilation
    rf_now = calc_rf(kernels, dilations, strides)
    if rf_now < target_rf:
        gap = target_rf - rf_now
        last_k = kernels[-1]
        last_d = dilations[-1]
        add_d = math.ceil(gap / (last_k - 1))
        dilations[-1] = min(d_max, last_d + add_d)

    return kernels, dilations, strides


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
    def forward(self, x):
        return self.body(x)

class VarConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k_size, dilation, downsample=False):
        super().__init__()
        pad = (k_size//2) * dilation
        self.conv = nn.Conv2d(in_c, out_c, k_size, 1, pad,
                              dilation=dilation, bias=False)
        self.bn   = nn.BatchNorm2d(out_c)
        self.act  = nn.GELU()
        self.pool = nn.MaxPool2d(2) if downsample else nn.Identity()

    def forward(self, x):
        return self.pool(self.act(self.bn(self.conv(x))))

class SimpleEncoder(nn.Module):
    """Three downsampling stages; output 128-channel feature map of size (H/8 × W/8)."""
    def __init__(self, in_c=3, base=32):
        super().__init__()
        self.stage1 = ConvBlock(in_c,  base)
        self.stage2 = ConvBlock(base,  base*2)
        self.stage3 = ConvBlock(base*2, base*4)   # =128
        self.out_channels = base*4
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)          # [B,128,H/8,W/8]

class PeriodEncoder(nn.Module):
    """
    Select (kernel, dilation) dynamically based on k; 3 downsampling stages → C_out=128.
    """
    def __init__(self, in_c, base_c, k, win_size, img_size,
                 depth=3, rf_ratio=1.6, p_pix_override=None):
        super().__init__()
        ks, ds, _ = period_to_conv_cfg_rfmatch(
            k, win_size, img_size,
            depth=depth, rf_ratio=rf_ratio,
            p_pix_override=p_pix_override
        )

        layers = []
        c_in = in_c
        for i in range(depth):
            c_out = base_c * (2**i)
            layers.append(
                VarConvBlock(c_in, c_out, ks[i], ds[i], downsample=True))
            c_in = c_out
        self.body = nn.Sequential(*layers)
        self.out_channels = c_in      # 128

    def forward(self, x):
        return self.body(x)           # [B,128,H/8,W/8]


class PeriodDecoder(nn.Module):
    """
    Symmetric upsampling; share kernel table with the encoder but in reverse order (dilation reversed as well).
    """
    def __init__(self, out_c, in_c, k, win_size, img_size,
                 depth=3, rf_ratio=1.6, p_pix_override=None):
        super().__init__()
        ks, ds, _ = period_to_conv_cfg_rfmatch(
            k, win_size, img_size,
            depth=depth, rf_ratio=rf_ratio,
            p_pix_override=p_pix_override
        )

        # Reverse ks and ds for decoding
        ks, ds = ks[::-1], ds[::-1]

        c_in = in_c          # 128
        layers = []
        for i in range(depth):
            c_mid = c_in // 2
            layers.append(nn.ConvTranspose2d(c_in, c_mid, 2, 2))  # up×2
            layers.append(nn.GELU())
            layers.append(
                nn.Conv2d(c_mid, c_mid, ks[i], 1,
                          (ks[i]//2)*ds[i], dilation=ds[i]))
            layers.append(nn.GELU())
            c_in = c_mid
        self.body  = nn.Sequential(*layers)
        self.final = nn.Conv2d(c_in, out_c, 1)
        self.img_size = img_size

    def forward(self, x):
        x = self.body(x)
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, (self.img_size, self.img_size),
                              mode='bilinear', align_corners=False)
        return self.final(x)

class MPJL(nn.Module):
    """
    global-encoder  +  K period-encoders
                 →   (Softmax-weighted fusion)
              fused_feat
                 →   K dedicated decoders
    """

    def __init__(self, n_channels=3, img_size=64, win_size=32, k_list=(10, 20, 30),
                 erf_gamma=1.6, erf_map_mode="aligned", erf_map_seed=0, erf_uniform_ref="max"):
        super().__init__()
        self.img_size  = img_size
        self.win_size = win_size
        self.k_list    = list(k_list)
        self.erf_gamma = float(erf_gamma)

        # Encoders
        self.global_encoder  = SimpleEncoder(n_channels)
        self.period_encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.erf_map_mode = erf_map_mode
        self.erf_map_seed = erf_map_seed
        self.erf_uniform_ref = erf_uniform_ref

        def _p_pix(k):
            return max(1, int(math.ceil(k / float(self.win_size) * float(self.img_size))))

        k_sorted = sorted(self.k_list)
        raw = {int(k): _p_pix(int(k)) for k in self.k_list}

        t_sorted = [raw[int(k)] for k in k_sorted]
        mode = self.erf_map_mode

        if mode == "aligned":
            t_used_sorted = list(t_sorted)

        elif mode == "inverse":
            t_used_sorted = list(t_sorted[::-1])

        elif mode == "random":
            rng = np.random.RandomState(self.erf_map_seed)
            t_used_sorted = list(t_sorted)
            rng.shuffle(t_used_sorted)

        elif mode == "uniform":
            if self.erf_uniform_ref == "min":
                k_ref = min(k_sorted)
            else:
                k_ref = max(k_sorted)
            t0 = raw[int(k_ref)]
            t_used_sorted = [t0 for _ in k_sorted]

        else:
            t_used_sorted = list(t_sorted)

        used = {int(k_sorted[i]): int(t_used_sorted[i]) for i in range(len(k_sorted))}

        # store as model attributes for later artifact saving
        self.erf_target_pix_raw = raw
        self.erf_target_pix_used = used

        print(f"[ERF-MAP] mode={mode} seed={self.erf_map_seed} uniform_ref={self.erf_uniform_ref}")
        print("[ERF-MAP] raw :", self.erf_target_pix_raw)
        print("[ERF-MAP] used:", self.erf_target_pix_used)

        for k in self.k_list:
            p_pix_used = self.erf_target_pix_used[int(k)]

            self.period_encoders[f'k_{k}'] = PeriodEncoder(
                in_c=n_channels, base_c=32,
                k=k, win_size=win_size, img_size=img_size,
                rf_ratio=self.erf_gamma,
                p_pix_override=p_pix_used
            )
            self.decoders[f'k_{k}'] = PeriodDecoder(
                out_c=n_channels, in_c=128,
                k=k, win_size=win_size, img_size=img_size,
                rf_ratio=self.erf_gamma,
                p_pix_override=p_pix_used
            )

        # learnable fusion weights (1+K)
        self.alpha = nn.Parameter(torch.zeros(len(self.k_list)+1))

        self.erf_map_mode = str(erf_map_mode)
        self.erf_map_seed = int(erf_map_seed)
        self.erf_uniform_ref = str(erf_uniform_ref)

    def _fuse_features(self, feats):
        """
        feats: list of feature maps [B,C,H,W] (length = 1 + K).
        Returns the Softmax-weighted sum across encoders.
        """
        w = torch.softmax(self.alpha, dim=0)      # [1+K]
        fused = sum(wi * fi for wi, fi in zip(w, feats))
        return fused                              # [B,C,H,W]

    # forward
    def forward(self, x, k):
        """
        x : input image tensor of shape [B,C,H,W]
        k : current period; must be one of self.k_list
        """
        if k not in self.k_list:
            raise ValueError(f'k={k} not in k_list {self.k_list}')

        # Extract features from all encoders
        feats = []
        feats.append(self.global_encoder(x))                 # global
        for k_each in self.k_list:
            feats.append(self.period_encoders[f'k_{k_each}'](x))

        # Fuse features
        fused_feat = self._fuse_features(feats)              # [B,128,H/8,W/8]

        # Decode with the period-specific decoder
        out = self.decoders[f'k_{k}'](fused_feat)
        return out