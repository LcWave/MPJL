import numpy as np
import os

import torch
from tqdm import tqdm
from pyts.image import GramianAngularField

from model.RevIN import RevIN
from utils.corrupt import inject_spikes

def generate_gaf_image(ts, image_size=None):
    """Convert a univariate time series to a GAF image."""
    gaf = GramianAngularField(image_size=image_size or len(ts), method='summation')
    ts = ts.reshape(1, -1)
    gaf_img = gaf.fit_transform(ts)[0]
    return gaf_img

def multivariate_to_gaf_image(multivar_ts, image_size=None):
    """Convert a multivariate time series segment to a multi-channel GAF image."""
    win_size, C = multivar_ts.shape
    channels = []
    for c in range(C):
        gaf_img = generate_gaf_image(multivar_ts[:, c], image_size)
        channels.append(gaf_img)
    return np.stack(channels, axis=0)  # shape: [C, H, W]

def extract_centered_segments(data, win_size):
    """Extract [T, win_size, C] segments centered at each timestamp with edge padding."""
    T, C = data.shape
    half = win_size // 2
    segments = []
    for t in range(T):
        start = t - half
        end = start + win_size
        if start < 0:
            pad = np.repeat(data[0:1], -start, axis=0)
            segment = np.concatenate([pad, data[0:end]], axis=0)
        elif end > T:
            pad = np.repeat(data[-1:], end - T, axis=0)
            segment = np.concatenate([data[start:T], pad], axis=0)
        else:
            segment = data[start:end]
        segments.append(segment)
    return np.stack(segments)  # shape: [T, win_size, C]

def minmax_normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def save_patch_dataset(npy_path, save_path, win_size=32, image_size=8, spike_cfg=None):
    data = np.load(npy_path)  # [T, C]
    data = np.nan_to_num(data)

    segments = extract_centered_segments(data, win_size=win_size)  # [T, win_size, C]
    patches = []

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    revin_layer = RevIN(num_features=segments.shape[2]).to(device)
    for seg in tqdm(segments, desc=f"Processing {os.path.basename(npy_path)}"):
        seg_tensor = torch.tensor(seg, dtype=torch.float32).to(device)  # [win_size, C]
        seg_tensor = revin_layer(seg_tensor, mode='norm')  # RevIN normalization
        seg_np = seg_tensor.cpu().numpy()

        gaf_img = multivariate_to_gaf_image(seg_np.astype(np.float32), image_size=image_size)  # [C, H, W]
        patches.append(gaf_img)
    patches = np.stack(patches)  # [T, C, H, W]
    np.save(save_path, patches)
    return patches.shape

def _spike_tag(cfg):
    if not cfg or (cfg.get("p", 0) <= 0):
        return ""
    pc = "pc" if cfg.get("per_channel", True) else "all"
    bp = cfg.get("burst_prob", 0)
    br = cfg.get("burst_len_range", (2,5))
    return f"_spike-p{cfg['p']}-A{cfg['A']}-bp{bp}-L{br[0]}-{br[1]}-{pc}"

def preprocess(dataset, root_dir, win_size, image_size, spike_cfg_train=None, spike_cfg_test=None):
    data_dir = os.path.join(root_dir, dataset)

    tag = _spike_tag(spike_cfg_train if spike_cfg_train else spike_cfg_test)

    output_dir = os.path.join(data_dir, 'patches', f'win{win_size}_img{image_size}{tag}')
    os.makedirs(output_dir, exist_ok=True)

    shape_train = save_patch_dataset(
        npy_path=os.path.join(data_dir, f'{dataset}_train.npy'),
        save_path=os.path.join(output_dir, f'{dataset}_train.npy'),
        win_size=win_size,
        image_size=image_size,
        spike_cfg=spike_cfg_train
    )

    shape_test = save_patch_dataset(
        npy_path=os.path.join(data_dir, f'{dataset}_test.npy'),
        save_path=os.path.join(output_dir, f'{dataset}_test.npy'),
        win_size=win_size,
        image_size=image_size,
        spike_cfg=spike_cfg_test
    )

    print("Train patch shape:", shape_train)
    print("Test patch shape:", shape_test)
