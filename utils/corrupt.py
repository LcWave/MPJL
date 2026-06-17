# utils/corrupt.py
import numpy as np

def _mad_scale(x, axis=0, eps=1e-8):
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True) + eps
    return 1.4826 * mad  # sigma ≈ 1.4826 * MAD

def inject_spikes(
    arr,                    # np.ndarray, shape [T, C] or [B, T, C]
    p=0.005,                # probability of a (time, channel) cell being selected as a spike
    A=5.0,                  # amplitude factor, multiplies the local scale
    burst_prob=0.3,         # probability of a burst occurring
    burst_len_range=(2, 5), # burst length range (inclusive)
    per_channel=True,       # True: sample each channel independently; False: same position across channels
    seed=42
):
    rng = np.random.default_rng(seed)
    x = arr.copy()
    orig_shape = x.shape

    if x.ndim == 2:
        x = x[None, ...]
    B, T, C = x.shape

    scale = _mad_scale(x, axis=1)  # [B,1,C]
    scale = np.maximum(scale, 1e-6)

    for b in range(B):
        if per_channel:
            mask = rng.random((T, C)) < p
        else:
            tmask = rng.random(T) < p
            mask = np.tile(tmask[:, None], (1, C))

        # randomly generate bursts
        if burst_prob and burst_prob > 0:
            t = 0
            while t < T:
                if rng.random() < burst_prob and (rng.random((1,)).item() < p * 2):
                    L = int(rng.integers(burst_len_range[0], burst_len_range[1] + 1))
                    t2 = min(T, t + L)
                    if per_channel:
                        cmask = rng.random(C) < 0.5
                        mask[t:t2, cmask] = True
                    else:
                        mask[t:t2, :] = True
                    t = t2
                else:
                    t += 1

        signs = rng.choice([-1.0, 1.0], size=(T, C))
        amp = signs * A * scale[b, 0, :]  # broadcast to [T, C]
        x[b] = np.where(mask, x[b] + amp, x[b])

    return x.reshape(orig_shape)
