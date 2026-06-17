from collections import defaultdict
import numpy as np
from astropy.stats import biweight_midvariance
from statsmodels.tsa.filters.hp_filter import hpfilter

from .modwt import modwt
from .huberacf import huber_acf, get_ACF_period


def extract_trend(y, reg):
    _, trend = hpfilter(y, reg)
    y_hat = y - trend
    return trend, y_hat


def huber_func(x, c):
    return np.sign(x) * np.minimum(np.abs(x), c)


# Use the true MAD (median absolute deviation); otherwise spurious p/2 periods are easily picked up.
def MAD(x, eps=1e-12):
    x = np.asarray(x, dtype=np.float64)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * mad + eps


def residual_autocov(x, c, eps=1e-12):
    mu = np.median(x)
    s = MAD(x, eps=eps)
    return huber_func((x - mu) / s, c)


def _top_peaks(acf_arr, min_p, max_p, topn=3, min_sep=2):
    """Pick top-N local maxima in ACF within [min_p, max_p] (no scipy)."""
    a = np.asarray(acf_arr, dtype=np.float64)
    lo = int(max(1, min_p))
    hi = int(min(max_p, len(a) - 1))
    if hi <= lo:
        return []

    seg = a[lo:hi + 1]
    idx = np.where((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
    if len(idx) == 0:
        idx = np.array([int(np.argmax(seg))])

    idx = idx[np.argsort(seg[idx])[::-1]]  # desc by peak height
    chosen = []
    for j in idx:
        p = lo + int(j)
        if all(abs(p - q) >= min_sep for q in chosen):
            chosen.append(p)
        if len(chosen) >= topn:
            break
    return chosen


def robust_period_full(
    x,
    wavelet_method='db4',
    num_wavelet=5,
    lmb=1600,
    c=1.5,
    zeta=1.345,
    # enforce period <= L (caller passes max_period=L)
    min_period=8,
    max_period=None,        # caller should pass L explicitly; defaults to n//2 if omitted
    # stability: keep multiple candidate periods per level instead of a single final_period
    peaks_per_level=3,
    keep_topk=5,
    # suppress domination by high-frequency levels
    skip_levels=1,
    lowfreq_penalty=True,
):
    assert wavelet_method.startswith('db')
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if max_period is None:
        max_period = n // 2

    # ---- HARD CLAMP: enforce bounds ----
    min_period = int(max(1, min_period))
    max_period = int(max(min_period, min(max_period, n // 2)))

    trend, y_hat = extract_trend(x, lmb)
    y_prime = residual_autocov(y_hat, c)

    W = modwt(y_prime, wavelet_method, level=num_wavelet)
    bivar = np.array([biweight_midvariance(w) for w in W])

    period_scores = defaultdict(float)
    ACF = []

    for i, w in enumerate(W):
        if i < int(skip_levels):
            continue

        try:
            acf = huber_acf(w, zeta=zeta)
        except TypeError:
            acf = huber_acf(w)

        acf2 = np.asarray(acf, dtype=np.float64).copy()

        # ---- enforce lag range [min_period, max_period] ----
        acf2[:min_period] = -np.inf
        if max_period + 1 < len(acf2):
            acf2[max_period + 1:] = -np.inf

        ACF.append(acf2)

        # ---- multi-peak accumulation ----
        min_sep = max(2, min_period // 10)
        cand = _top_peaks(acf2, min_period, max_period,
                          topn=int(peaks_per_level), min_sep=min_sep)

        # fallback to get_ACF_period but still obey bounds
        if len(cand) == 0:
            try:
                _, p0, _ = get_ACF_period(acf2)
                if p0 is not None:
                    p0 = int(p0)
                    if min_period <= p0 <= max_period:
                        cand = [p0]
            except Exception:
                cand = []

        if len(cand) == 0:
            continue

        wgt = float(bivar[i])
        if lowfreq_penalty:
            wgt = wgt / float(i + 1)

        for p in cand:
            p = int(p)
            if min_period <= p <= max_period:
                period_scores[p] += wgt

    # ---- avoid empty / avoid harsh filtering ----
    if len(period_scores) == 0:
        periods = np.array([min_period], dtype=int)
        return periods, W, bivar, ACF, None, ACF

    items = sorted(period_scores.items(), key=lambda kv: -kv[1])
    items = items[:max(1, int(keep_topk))]
    periods = np.array([int(p) for p, _ in items], dtype=int)

    return periods, W, bivar, ACF, None, ACF