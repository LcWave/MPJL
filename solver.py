import csv
import os
from statsmodels.tsa.stattools import acf
import tqdm

from model.model import MPJL

from utils.utils import *
from data_factory.data_loader import get_loader_segment
from period.robustperiod import robust_period_full
from metrics.metrics import *
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

try:
    from scipy.stats import genpareto
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _binary_to_runs(x):
    x = np.asarray(x, dtype=int).ravel()
    pad = np.pad(x, (1,1), constant_values=0)
    d = np.diff(pad)
    starts = np.where(d == 1)[0]
    ends   = np.where(d == -1)[0]
    return [(int(s), int(e)) for s, e in zip(starts, ends)]

def _intersections(runs_a, runs_b):
    out = []
    ia = ib = 0
    while ia < len(runs_a) and ib < len(runs_b):
        sa, ea = runs_a[ia]; sb, eb = runs_b[ib]
        s = max(sa, sb); e = min(ea, eb)
        if s < e: out.append((s, e))
        if ea <= eb: ia += 1
        else: ib += 1
    return out

def plot_raw_center_with_labels(raw_1d, gt, pred, win_size, *,
                                stride=1, base_start=0,   # base_start: index of sample window 0 in the raw sequence (default 0)
                                start=None, end=None,     # sample-index range [start, end) to plot (sample indices, not raw indices)
                                save_path=None, title='Raw Sensor with GT & Pred'):
    """
    raw_1d: raw 1D sensor sequence (length T)
    gt/pred: sample-level 0/1 sequences for the test set (length N, same length as pred/gt)
    win_size: length of the sample window (used to map sample indices to raw centers)
    stride: sliding-window stride (default 1)
    base_start: offset of test sample window 0 within the raw sequence (set if the test set does not start at 0)
    start/end: plot only a sub-range of sample indices (e.g., 5500-6100)
    """
    gt = np.asarray(gt, dtype=int).ravel()
    pred = np.asarray(pred, dtype=int).ravel()
    N = len(gt)
    assert len(pred) == N, "gt/pred length mismatch"

    raw_1d = np.asarray(raw_1d).astype(float).ravel()
    T = len(raw_1d)

    # raw-sequence index of the center of sample i
    centers = base_start + (np.arange(N) * stride) + (win_size // 2)
    valid = (centers >= 0) & (centers < T)
    if not np.all(valid):
        # truncate to the valid range
        first = int(np.argmax(valid))
        last  = int(len(valid) - np.argmax(valid[::-1]))
        gt, pred, centers = gt[first:last], pred[first:last], centers[first:last]
        N = len(gt)

    # use the raw value at each center as the plotted curve
    series = raw_1d[centers]

    s = 0 if start is None else max(0, int(start))
    e = N if end   is None else min(N, int(end))

    xs = centers[s:e]              # x-axis uses raw indices
    ys = series[s:e]
    y_min, y_max = float(np.nanmin(ys)), float(np.nanmax(ys))
    pad = 0.05 * (y_max - y_min + 1e-12)
    y_min -= pad; y_max += pad

    # event intervals (in sample-index space)
    gt_runs_all   = _binary_to_runs(gt)
    pred_runs_all = _binary_to_runs(pred)
    ovl_runs_all  = _intersections(gt_runs_all, pred_runs_all)

    # map sample-index intervals to raw-index intervals and clip
    def _runs_to_raw(runs, s, e, centers):
        out = []
        for a, b in runs:
            if b <= s or a >= e:
                continue
            a = max(a, s); b = min(b, e)
            # [a, b) -> raw indices [centers[a], centers[b-1] + 1)
            xa = int(centers[a])
            xb = int(centers[b-1]) + 1
            if xb > xa:
                out.append((xa, xb))
        return out

    gt_raw   = _runs_to_raw(gt_runs_all, s, e, centers)
    pred_raw = _runs_to_raw(pred_runs_all, s, e, centers)
    ovl_raw  = _runs_to_raw(ovl_runs_all, s, e, centers)

    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(xs, ys, linewidth=1.0, label='Raw (channel center)')

    drawn = set()
    for a, b in ovl_raw:
        rect = Rectangle((a, y_min), b-a, y_max-y_min, facecolor='purple', alpha=0.25, lw=0)
        ax.add_patch(rect)
        if 'Overlap' not in drawn:
            rect.set_label('Overlap (GT ∩ Pred)'); drawn.add('Overlap')
    for a, b in gt_raw:
        rect = Rectangle((a, y_min), b-a, y_max-y_min, facecolor='tab:red', alpha=0.18, lw=0)
        ax.add_patch(rect)
        if 'GT' not in drawn:
            rect.set_label('GT'); drawn.add('GT')
    for a, b in pred_raw:
        rect = Rectangle((a, y_min), b-a, y_max-y_min, facecolor='tab:blue', alpha=0.18, lw=0)
        ax.add_patch(rect)
        if 'Pred' not in drawn:
            rect.set_label('Pred'); drawn.add('Pred')

    ax.set_xlim(int(xs[0]), int(xs[-1]))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('Raw Time Index')
    ax.set_ylabel('Sensor Value')
    ax.set_title(f'{title}  (win={win_size}, stride={stride})  [samples {s}:{e})')
    ax.legend(ncol=4, frameon=True)
    ax.grid(alpha=0.15, linestyle=':')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200)
        plt.close(fig)
    else:
        plt.show()



def norm(x):
    """
        Normalize along the last dimension with a small epsilon to avoid division by zero.
        """
    mean = torch.mean(x, dim=-1, keepdim=True)
    stddev = torch.std(x, dim=-1, keepdim=True)

    normalized_tensor = (x - mean) / (stddev + 1e-5)
    return  normalized_tensor



def adjust_learning_rate(shared_opt, period_opts, epoch, base_lr):
    """
        Cosine-like step decay (here: halve every epoch as written).
        Apply the same LR to the shared optimizer and all per-period optimizers.
        """
    lr = base_lr * (0.5 ** ((epoch-1)//1))
    for pg in shared_opt.param_groups:
        pg['lr'] = lr
    for opt in period_opts.values():
        for pg in opt.param_groups:
            pg['lr'] = lr

# def margin_loss(x_rec: torch.Tensor,
#                 x_ref: torch.Tensor,
#                 x_in : torch.Tensor,
#                 lam: float = 1.0,
#                 margin: float = 0.05):
#     """
#     Push–Pull bilateral loss:
#       (1) Pull reconstructed output toward the normal reference.
#       (2) Push it away from the input; penalize if the distance is below `margin`.
#     """
#     pull = F.mse_loss(x_rec, x_ref)                         # scalar
#     push = F.mse_loss(x_rec, x_in, reduction='none')        # [B,C,H,W]
#     push = push.flatten(1).mean(1)                          # [B]
#     push = lam * F.relu(margin - push).mean()               # scalar
#     return pull + push

def margin_loss(
    x_rec: torch.Tensor,
    x_ref: torch.Tensor,
    x_in:  torch.Tensor,
    lam: float = 1.0,
    margin: float = 0.05,
    drift_tol: float = 0.01,   # delta: tolerance threshold below which the reference is considered drift-free
):
    """
    Drift-aware Push-Pull loss.

    x_rec : predicted output    (hat I_t^{(p)})
    x_ref : period-shifted reference frame (R_t^{(p)})
    x_in  : current input       (I_t)

    pull  = ||x_rec - x_ref||^2
    push  = lambda * g * max(0, m - ||x_rec - x_in||^2)

    where g = min(1, Delta / delta), Delta = ||x_ref - x_in||^2.
    Under a perfect period (x_ref ~= x_in), Delta ~ 0 => g ~ 0, so Push is automatically gated off.
    """

    # -------- Pull: draw the prediction toward the future reference --------
    # standard MSE averaged over (B, C, H, W) -> scalar
    pull = F.mse_loss(x_rec, x_ref)  # scalar

    # -------- Push: push the prediction away from the current input --------
    # per-sample distance d_rec_in(b) = MSE(x_rec[b], x_in[b])
    d_rec_in = F.mse_loss(x_rec, x_in, reduction='none')   # [B,C,H,W]
    d_rec_in = d_rec_in.flatten(1).mean(1)                 # [B]

    # raw margin-based push (larger penalty the closer the prediction is to the input)
    push_raw = F.relu(margin - d_rec_in)                   # [B], negative values clamped to 0

    # -------- Drift gate: measure how different the reference is from the current input --------
    # Delta(b) = MSE(x_ref[b], x_in[b])
    d_ref_in = F.mse_loss(x_ref, x_in, reduction='none')   # [B,C,H,W]
    d_ref_in = d_ref_in.flatten(1).mean(1)                 # [B]

    # g(b) = min(1, Delta/delta): g -> 0 when Delta is small, g -> 1 when Delta is large
    gate = torch.clamp(d_ref_in / (drift_tol + 1e-8), max=1.0)  # [B]

    # drift-gated push: the penalty is only active when non-negligible drift is present
    push = lam * (gate * push_raw).mean()                  # scalar

    return pull + push

class Solver(object):
    DEFAULTS = {}

    def __init__(self, config):

        self.__dict__.update(Solver.DEFAULTS, **config)

        # self.setup_logging()

        _dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
        self.train_loader = get_loader_segment(self.index, os.path.join(_dataset_root, self.data_path), batch_size=self.batch_size,
                                               win_size=self.win_size, image_size=self.input_size, mode='train', dataset=self.dataset)
        self.vali_loader = get_loader_segment(self.index, os.path.join(_dataset_root, self.data_path), batch_size=self.batch_size,
                                              win_size=self.win_size, image_size=self.input_size, mode='val', dataset=self.dataset)
        self.test_loader = get_loader_segment(self.index, os.path.join(_dataset_root, self.data_path), batch_size=self.batch_size,
                                              win_size=self.win_size, image_size=self.input_size, mode='test', dataset=self.dataset)
        self.thre_loader = get_loader_segment(self.index, os.path.join(_dataset_root, self.data_path), batch_size=self.batch_size,
                                              win_size=self.win_size, image_size=self.input_size, mode='thre', dataset=self.dataset)

        # self.revin_layer = RevIN(num_features=self.input_c)

        # self.build_model()

        data_path = os.path.join(_dataset_root, self.data_path)
        raw_train_path = os.path.join(data_path, f'{self.dataset}_train.npy')
        raw_train_data = np.load(raw_train_path)  # [T, C] numpy array
        raw_train_data = np.nan_to_num(raw_train_data)

        period_source = str(getattr(self, "period_source", "rwmpd"))
        self.period_source = period_source  # recorded for later use in artifacts/config.json

        if period_source == "fixed":
            k_in = getattr(self, "k_list", [])
            self.k_list_input = [int(x) for x in k_in]  # keep the raw input list
            if len(self.k_list_input) == 0:
                raise ValueError("period_source=fixed but --k_list is empty. "
                                 "Example: --k_list 2,4,8,16,32")
            self.periods = self.k_list_input
            self.periods_raw = list(self.periods)
            self.periods_dedup = list(self.periods)
            self.periods_mainharm = list(self.periods)
        else:
            self.k_list_input = None
            self.periods = self.estimate_periods_robust(
                raw_train_data,
                proxy_mode=str(getattr(self, "proxy_mode", "var")),
                proxy_top_m=int(getattr(self, "proxy_top_m", 1)),
            )
            self.periods_raw = list(self.periods)
            print("[RWMPD] raw periods:", self.periods_raw)

            dedup = self.dedup_periods_close(self.periods_raw, rel_tol=0.05, abs_tol=2)
            self.periods_dedup = list(dedup)
            self.periods = self.keep_main_plus_harmonics(
                self.periods_dedup,
                win_size=int(self.win_size),
                min_p=max(8, int(self.win_size // 16)),
                max_keep=3
            )
            self.periods_mainharm = list(self.periods)
            print("[RWMPD] dedup -> main+harmonics:", self.periods_mainharm)

        self.periods = self.apply_period_perturb(self.periods)
        self.periods_final = list(self.periods)
        print(
            f"[period_source] {period_source} | k_list_input={self.k_list_input} | periods(after perturb)={self.periods_final}")

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if self.model_name == 'MPJL':
            self.model = MPJL(
                n_channels=self.input_c,
                img_size=self.input_size,
                win_size=self.win_size,
                k_list=self.periods,
                erf_gamma=float(getattr(self, "erf_gamma", 1.6)),
                erf_map_mode=str(getattr(self, "erf_map_mode", "aligned")),
                erf_map_seed=int(getattr(self, "erf_map_seed", 0)),
                erf_uniform_ref=str(getattr(self, "erf_uniform_ref", "max")),
            ).to(self.device)

        def _count_params(m: torch.nn.Module):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        total_p, train_p = _count_params(self.model)
        print(f"[MODEL] params_total={total_p:,}  params_trainable={train_p:,}")
        self.model_param_stats = {"params_total": int(total_p), "params_trainable": int(train_p)}

        flops = self.try_profile_flops()
        if flops is not None:
            print(
                f"[MODEL] FLOPs approx (k={flops['k_profiled']}): flops={flops['flops']:.3e}  macs={flops['macs']:.3e}")
            self.model_flops_stats = flops

        if self.loss_fuc == 'MSE':
            self.criterion = nn.MSELoss()
        elif self.loss_fuc == 'MAE':
            self.criterion = nn.L1Loss()
        elif self.loss_fuc == 'margin':
            self.criterion = None

        self.shared_optimizer = torch.optim.Adam(
            list(self.model.global_encoder.parameters()) +
            [self.model.alpha],  # fusion weights
            lr=self.lr)

        self.period_optim = {}  # dict to store per-period optimizers

    def try_profile_flops(self):
        # Wrap model(x,k) into a module that only takes x (for profiling tools)
        class _Wrap(nn.Module):
            def __init__(self, m, k):
                super().__init__()
                self.m = m
                self.k = int(k)

            def forward(self, x):
                return self.m(x, self.k)

        k0 = int(self.periods[0]) if hasattr(self, "periods") and len(self.periods) > 0 else 1
        x = torch.zeros(1, int(self.input_c), int(self.input_size), int(self.input_size), device=self.device)

        # 1) try thop
        try:
            from thop import profile
            wrap = _Wrap(self.model, k0).to(self.device)
            macs, params = profile(wrap, inputs=(x,), verbose=False)
            # FLOPs often counted as 2*MACs
            flops = 2.0 * float(macs)
            return {"k_profiled": k0, "macs": float(macs), "flops": float(flops), "tool": "thop"}
        except Exception as e:
            print("[MODEL] FLOPs profiling skipped (thop not available):", str(e))
            return None

    def pixel_pooling(self, loss_map: torch.Tensor) -> torch.Tensor:
        """
        loss_map: [B,C,H,W] (MSE reduction='none')
        return : [B]
        """
        # -------- stress test knobs (no-op by default) --------
        pool_scale = float(getattr(self, "pool_scale", 1.0))
        if abs(pool_scale - 1.0) > 1e-6:
            loss_map = F.interpolate(loss_map, scale_factor=pool_scale, mode="nearest")

        sparse_keep = float(getattr(self, "sparse_keep", 1.0))
        if sparse_keep < 1.0:
            B = loss_map.size(0)
            flat = loss_map.reshape(B, -1)
            med = flat.median(dim=1, keepdim=True)[0]
            N = flat.size(1)
            m = max(1, int(round(sparse_keep * N)))
            idx = torch.rand(B, N, device=flat.device).argsort(dim=1)[:, :m]
            mask = torch.zeros_like(flat, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            flat2 = med.expand_as(flat).clone()
            flat2[mask] = flat[mask]
            loss_map = flat2.reshape_as(loss_map)

        mode = str(getattr(self, "pixel_pool", "min"))
        x = loss_map
        B = x.size(0)
        x = x.reshape(B, -1)  # [B, N]
        N = x.size(1)

        if mode == "min":
            return x.min(dim=1)[0]

        if mode.startswith("q"):  # q05 / q10 ...
            # e.g., pixel_pool="q10"
            q = float(mode[1:]) / 100.0
            return torch.quantile(x, q, dim=1)

        if mode.startswith("topkmean"):  # topkmean_5  -> 5%
            r = float(mode.split("_")[1]) / 100.0
            k = max(1, int(round(r * N)))
            vals = torch.topk(x, k=k, dim=1, largest=False).values
            return vals.mean(dim=1)

        if mode.startswith("trim"):  # trim_5 -> drop lowest 5%
            r = float(mode.split("_")[1]) / 100.0
            k = int(round(r * N))
            if k <= 0:
                return x.mean(dim=1)
            xs, _ = torch.sort(x, dim=1)
            xs = xs[:, k:]  # drop lowest tail
            return xs.mean(dim=1)

        if mode.startswith("region"):  # region_q10 / region_topkmean_10
            # first take the regional minimum, then pool over the region values
            g = int(getattr(self, "region_grid", 4))
            # adaptive max on (-x) gives region-wise min
            # [B,C,H,W] -> [B,C,g,g]
            u = -F.adaptive_max_pool2d(-loss_map, output_size=(g, g))
            u = u.reshape(B, -1)  # [B, C*g*g]

            sub = mode.split("_", 1)[1] if "_" in mode else "q10"

            if sub.startswith("q"):
                q = float(sub[1:]) / 100.0
                return torch.quantile(u, q, dim=1)

            if sub.startswith("topkmean"):
                r = float(sub.split("_")[1]) / 100.0
                k = max(1, int(round(r * u.size(1))))
                vals = torch.topk(u, k=k, dim=1, largest=False).values
                return vals.mean(dim=1)

            # fallback
            return torch.quantile(u, 0.10, dim=1)

        # fallback
        return x.min(dim=1)[0]

    def expert_pooling(self, s_mat: torch.Tensor) -> torch.Tensor:
        """
        s_mat: [B, K] per-expert scalar scores
        return: [B]
        """
        mode = str(getattr(self, "expert_pool", "min"))
        if mode == "min":
            return s_mat.min(dim=1)[0]
        if mode == "mean":
            return s_mat.mean(dim=1)
        if mode == "max":
            return s_mat.max(dim=1)[0]
        if mode.startswith("q"):  # q40
            q = float(mode[1:]) / 100.0
            return torch.quantile(s_mat, q, dim=1)
        if mode.startswith("topkmean"):  # topkmean_2 -> avg of smallest 2 experts
            k = int(mode.split("_")[1])
            k = max(1, min(k, s_mat.size(1)))
            vals = torch.topk(s_mat, k=k, dim=1, largest=False).values
            return vals.mean(dim=1)
        return s_mat.min(dim=1)[0]

    def pot_threshold(self, train_scores: np.ndarray, target_rate: float,
                      q: float = 0.98, min_exc: int = 50):
        """
        POT/EVT thresholding using GPD fit on exceedances.

        train_scores: training anomaly scores (higher => more anomalous)
        target_rate : desired alarm rate on test (e.g., 0.003 for 0.3%)
        q           : base threshold quantile u on train
        min_exc     : minimum exceedances to fit GPD; else fallback to percentile
        """
        x = np.asarray(train_scores, dtype=np.float64).ravel()
        x = x[np.isfinite(x)]
        if len(x) < 10:
            # degenerate fallback
            return float(np.quantile(x, 1 - target_rate))

        # base threshold u
        u = float(np.quantile(x, q))
        exc = x[x > u] - u  # exceedances
        p_u = len(exc) / (len(x) + 1e-12)

        # if exceedances too few or scipy missing -> fallback to simple percentile
        if (not _HAS_SCIPY) or (len(exc) < min_exc) or (p_u <= 0):
            return float(np.quantile(x, 1 - target_rate))

        # Fit GPD on exceedances (loc fixed to 0)
        try:
            xi, loc, beta = genpareto.fit(exc, floc=0.0)  # xi=shape, beta=scale
            beta = float(beta)
            xi = float(xi)
        except Exception:
            return float(np.quantile(x, 1 - target_rate))

        # EVT threshold formula
        # target_rate = P(X > thr)
        # p_u = P(X > u) on train
        # For xi != 0: thr = u + beta/xi * ((p_u/target_rate)**xi - 1)
        # For xi ~ 0: thr = u + beta * log(p_u/target_rate)
        target_rate = max(target_rate, 1e-12)
        ratio = p_u / target_rate
        ratio = max(ratio, 1e-12)

        if abs(xi) < 1e-6:
            thr = u + beta * np.log(ratio)
        else:
            thr = u + (beta / xi) * (ratio ** xi - 1.0)

        # numeric safety
        thr = float(max(thr, u))
        return thr

    def _get_ref_index(self, base_idx, k, total_len):
        mode = str(getattr(self, "psp_wrap", "wrap"))
        ref = base_idx + k
        if mode == "wrap":
            return ref % total_len
        # drop / no-wrap
        if ref >= total_len:
            return None
        return ref

    def generate_reference_patch(self, index, k, source='train'):
        """
        Fetch the GAF patch at position (index + k) with wrap-around at sequence boundaries.
        """
        if source == 'train':
            dataset = self.train_loader.dataset
        elif source == 'val':
            dataset = self.vali_loader.dataset
        elif source == 'test':
            dataset = self.test_loader.dataset
        else:
            raise ValueError(f"Unsupported source: {source}")

        total_len = len(dataset)
        # calculate reference index
        ref_index = (index + k) % total_len

        patch, _ = dataset[ref_index]  # [C, H, W]
        return patch

    def linear_detrend(self, y):
        """
        Remove linear trend from the input signal.
        """
        n = len(y)
        t = np.arange(n)
        A = np.vstack([t, np.ones(n)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        return y - (m * t + c)

    @staticmethod
    def _mad(x, eps=1e-12):
        x = np.asarray(x, dtype=np.float64)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return 1.4826 * mad + eps

    def select_scalar_proxy(self, series, mode="var", top_m=1):
        x = np.asarray(series, dtype=np.float64)

        if x.ndim == 1:
            info = {"mode": "single", "selected_channels": [0], "weights": [1.0]}
            return x.astype(np.float64), info

        T, C = x.shape
        x_center = x - x.mean(axis=0, keepdims=True)
        chan_var = x_center.var(axis=0)  # reused across multiple modes

        # -------- var / top-m variance aggregation --------
        if mode == "var":
            idx = np.argsort(chan_var)[::-1][:max(1, int(top_m))]
            w = chan_var[idx].copy()
            if w.sum() <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / (w.sum() + 1e-12)

            r = x_center[:, idx] @ w  # combining centered channels is more stable
            info = {
                "mode": "var",
                "top_m": int(top_m),
                "selected_channels": idx.tolist(),
                "channel_var": chan_var[idx].tolist(),
                "weights": w.tolist(),
            }
            return r.astype(np.float64), info

        # -------- PCA first component --------
        if mode == "pca":
            cov = np.cov(x_center, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, -1]
            sign = np.sign(v.sum()) or 1.0
            v = v * sign
            r = x_center @ v
            info = {
                "mode": "pca",
                "weights": v.tolist(),
                "explained_var_ratio": float(eigvals[-1] / (eigvals.sum() + 1e-12)),
            }
            return r.astype(np.float64), info

        # -------- Robust PCA (median/MAD + Huber clip + PCA) --------
        if mode == "rpca":
            clip = float(getattr(self, "proxy_rpca_clip", 3.0))

            # robust standardize per channel
            med = np.median(x, axis=0, keepdims=True)
            mad = np.array([self._mad(x[:, j]) for j in range(C)], dtype=np.float64).reshape(1, C)
            x_rob = (x - med) / mad
            x_rob = np.clip(x_rob, -clip, clip)

            # optionally restrict RPCA to the top_m most consistently active channels
            m = max(1, int(top_m))
            if m < C:
                # select channels by their robust (post-clipping) variance
                rv = x_rob.var(axis=0)
                idx = np.argsort(rv)[::-1][:m]
                x_use = x_rob[:, idx]
            else:
                idx = np.arange(C)
                x_use = x_rob

            cov = np.cov(x_use, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, -1]
            sign = np.sign(v.sum()) or 1.0
            v = v * sign

            r = x_use @ v
            info = {
                "mode": "rpca",
                "top_m": int(top_m),
                "selected_channels": idx.tolist(),
                "clip": clip,
                "weights": v.tolist(),
                "explained_var_ratio": float(eigvals[-1] / (eigvals.sum() + 1e-12)),
            }
            return r.astype(np.float64), info

        # -------- Multi-channel energy (L2 norm over channels) --------
        if mode == "energy":
            m = max(1, int(top_m))
            if m < C:
                idx = np.argsort(chan_var)[::-1][:m]
                x_use = x_center[:, idx]
            else:
                idx = np.arange(C)
                x_use = x_center
            r = np.sqrt((x_use ** 2).sum(axis=1) + 1e-12)
            info = {"mode": "energy", "top_m": int(top_m), "selected_channels": idx.tolist()}
            return r.astype(np.float64), info

        # -------- Energy-weighted (variance-weighted L2 energy) --------
        if mode == "energyw":
            m = max(1, int(top_m))
            if m < C:
                idx = np.argsort(chan_var)[::-1][:m]
            else:
                idx = np.arange(C)

            w = chan_var[idx].copy()
            if w.sum() <= 0:
                w = np.ones_like(w) / len(w)
            else:
                w = w / (w.sum() + 1e-12)

            x_use = x_center[:, idx]
            r = np.sqrt((x_use ** 2) @ w + 1e-12)
            info = {"mode": "energyw", "top_m": int(top_m),
                    "selected_channels": idx.tolist(), "weights": w.tolist()}
            return r.astype(np.float64), info

        raise ValueError(f"Unknown proxy selection mode: {mode}")

    def dedup_periods_close(self, periods, rel_tol=0.05, abs_tol=2):
        """
        Merge nearby periods into a single representative value (the median).
        rel_tol: relative tolerance (e.g., 5%)
        abs_tol: absolute tolerance (e.g., +/-2)
        """
        ps = sorted(int(p) for p in periods if int(p) > 0)
        if not ps:
            return []

        clusters = [[ps[0]]]
        for p in ps[1:]:
            rep = int(round(np.median(clusters[-1])))
            if abs(p - rep) <= abs_tol or abs(p - rep) <= int(round(rel_tol * max(rep, 1))):
                clusters[-1].append(p)
            else:
                clusters.append([p])

        reps = [int(round(np.median(c))) for c in clusters]
        return reps  # not truncated yet; the caller decides how many to keep

    def keep_main_plus_harmonics(self, periods, win_size, min_p=8, max_keep=3):
        """
        Keep only the main period plus its harmonics (p/2, p/3).
        - periods: deduplicated list (ascending or descending, either is fine)
        - win_size: L, used as the upper bound (a period should not exceed L)
        """
        if not periods:
            return []

        # main period: take the largest representative value (typically closer to the true mechanical cycle)
        p0 = int(max(periods))

        # clamp to a valid range
        p0 = max(min_p, min(p0, int(win_size)))

        out = [p0]

        # add harmonics: p/2, p/3
        for d in [2, 3]:
            q = int(round(p0 / d))
            if min_p <= q <= int(win_size) and q not in out:
                out.append(q)

        # order-preserving dedup + truncation
        out = out[:max_keep]
        return out

    def estimate_periods_robust(self,
                                series,
                                wavelet='db10',
                                num_wavelets=8,
                                lmb=1e6,
                                c=2,
                                zeta=1.345,
                                proxy_mode: str = "var",
                                proxy_top_m: int = 1):
        """
        Estimate multiple periods using RobustPeriod.
        Hyperparameters can be tuned for robustness and data characteristics.
        """
        series = np.asarray(series)

        # 1) adaptively select a 1D proxy series r_t
        proxy_1d, proxy_info = self.select_scalar_proxy(
            series,
            mode=proxy_mode,
            top_m=proxy_top_m
        )
        self.period_proxy_info = proxy_info  # optional: keep the metadata for later inspection

        # 2) linear detrending
        proxy_1d = self.linear_detrend(proxy_1d)

        # 3) call robust_period_full
        L = int(self.win_size)

        min_p = max(8, L // 16)  # e.g., 128 -> 8
        max_p = L // 2

        periods, *_ = robust_period_full(
            proxy_1d,
            wavelet_method=wavelet,
            num_wavelet=num_wavelets,
            lmb=lmb,
            c=c,
            zeta=zeta,
            min_period=min_p,
            max_period=max_p,
            peaks_per_level=3,
            keep_topk=5,
            skip_levels=1,
            lowfreq_penalty=True,
        )

        print("[RWMPD] raw periods:", periods)

        return [int(round(p)) for p in periods]

    def apply_period_perturb(self, periods):
        periods = [int(p) for p in periods if int(p) > 0]
        if len(periods) == 0:
            return periods

        mode = str(getattr(self, "period_perturb", "none"))
        rng = np.random.RandomState(int(getattr(self, "random_seed", 0)))

        if mode == "none":
            return periods

        if mode == "topk":
            k = int(getattr(self, "period_topk", len(periods)))
            k = max(1, min(k, len(periods)))
            return periods[:k]

        if mode == "drop_top1":
            return periods[1:] if len(periods) > 1 else periods

        if mode == "drop_tail1":
            return periods[:-1] if len(periods) > 1 else periods

        if mode == "jitter":
            ratio = float(getattr(self, "period_jitter_ratio", 0.05))
            out = []
            for p in periods:
                eps = rng.uniform(-ratio, ratio)
                pj = int(max(1, round(p * (1.0 + eps))))
                out.append(pj)
            # order-preserving dedup
            seen = set()
            out2 = []
            for p in out:
                if p not in seen:
                    out2.append(p);
                    seen.add(p)
            return out2

        return periods

    def debug_nans(self, tensor, name, where=""):
        """Print the first NaN position & abort the run."""
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            idx = torch.nonzero(~torch.isfinite(tensor))[0]
            print(f"\n[NaN‑Detect] {where} -> {name} has NaN/Inf @ index {idx} "
                  f"val={tensor.flatten()[idx]}")
            raise RuntimeError("NaN detected – aborting for debug!")

    def train(self):
        os.makedirs('weights', exist_ok=True)
        best_loss = float('inf')

        no_improve_count = 0

        print("periods:", self.periods)

        seed = int(getattr(self, "random_seed", 0))
        tag = str(getattr(self, "exp_tag", "default"))
        w_prefix = f"weights/{self.dataset}_{self.win_size}_{self.input_size}_{tag}_seed{seed}"
        best_path = w_prefix + "_best.pt"
        last_path = w_prefix + "_last.pt"

        with open(f'weights/{self.dataset}_step.csv', 'w') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss'])
            writer.writeheader()

            for epoch in range(self.num_epochs):
                self.model.train()
                epoch_loss = 0.0
                count_valid = 0.0

                p_bar = tqdm.tqdm(self.train_loader, total=len(self.train_loader))
                for global_index, (x_ori, _, idxs) in enumerate(p_bar):
                    x_ori = x_ori.to(self.device).float()  # [B, C, H, W]
                    # x_ori = self.revin_layer(x_ori, 'norm')

                    self.shared_optimizer.zero_grad()

                    for k in self.periods:
                        if k not in self.period_optim:
                            enc_k = self.model.period_encoders[f'k_{k}'].parameters()
                            dec_k = self.model.decoders[f'k_{k}'].parameters()
                            self.period_optim[k] = torch.optim.Adam(
                                list(enc_k) + list(dec_k), lr=self.lr)

                        self.period_optim[k].zero_grad()

                        # self.optimizer.zero_grad()
                        # Prepare reference image for this k
                        # rs_list = []
                        # dataset = self.train_loader.dataset.base
                        # total_len = len(dataset)
                        # for b in range(x_ori.size(0)):
                        #     base_idx = int(idxs[b])
                        #     ref_index = (base_idx + k) % total_len
                        #     rs, _ = dataset[ref_index]
                        #     rs_list.append(rs)
                        #
                        #
                        # x_ref = torch.tensor(np.stack(rs_list), dtype=torch.float32).to(self.device) # [B, C, H, W]
                        #
                        # pred = self.model(x_ori, k)

                        valid_bs = []
                        rs_list = []
                        dataset = self.train_loader.dataset.base
                        total_len = len(dataset)

                        for b in range(x_ori.size(0)):
                            base_idx = int(idxs[b])
                            ref_index = self._get_ref_index(base_idx, k, total_len)
                            if ref_index is None:
                                continue
                            rs, _ = dataset[ref_index]
                            rs_list.append(rs)
                            valid_bs.append(b)

                        if len(valid_bs) == 0:
                            continue

                        x_in = x_ori[valid_bs]  # keep only the valid samples
                        x_ref = torch.tensor(np.stack(rs_list), dtype=torch.float32).to(self.device)

                        pred = self.model(x_in, k)

                        self.debug_nans(pred, "pred", f"epoch{epoch}_step{global_index}_k{k}")
                        self.debug_nans(x_ref, "x_ref", f"epoch{epoch}_step{global_index}_k{k}")

                        # Loss
                        if self.loss_fuc == 'margin':
                            loss = margin_loss(pred, x_ref, x_in)
                        else:
                            loss = self.criterion(pred, x_ref)
                        self.debug_nans(loss, "loss", f"epoch{epoch}_step{global_index}_k{k}")

                        if torch.isnan(loss):
                            print('NaN loss detected, skipping step')
                            continue

                        # Backward and update only this decoder
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model.period_encoders[f'k_{k}'].parameters(), 1.0)
                        torch.nn.utils.clip_grad_norm_(self.model.decoders[f'k_{k}'].parameters(), 1.0)

                        self.period_optim[k].step()

                        # epoch_loss += loss.item() * x_ori.size(0)
                        epoch_loss += loss.item() * len(valid_bs)
                        count_valid += len(valid_bs)

                    torch.nn.utils.clip_grad_norm_(self.model.global_encoder.parameters(), 1.0)
                    torch.nn.utils.clip_grad_norm_([self.model.alpha], 1.0)
                    self.shared_optimizer.step()

                    p_bar.set_description(f"Epoch {epoch + 1} | Step Loss: {loss.item():.4f}")

                avg_loss = epoch_loss / (count_valid * len(self.periods) + 1e-12)
                writer.writerow({'epoch': epoch + 1, 'train_loss': f'{avg_loss:.6f}'})

                if avg_loss < best_loss:
                    best_loss = avg_loss
                    no_improve_count = 0
                    torch.save(self.model.state_dict(), best_path)
                else:
                    no_improve_count += 1
                    if no_improve_count >= self.patience:
                        print(f"Early stopping at epoch {epoch + 1}, no improvement for {self.patience} epochs.")
                        break

                print(f"Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f} | Best So Far: {best_loss:.4f}")

                torch.save(self.model.state_dict(), last_path)

                adjust_learning_rate(self.shared_optimizer, self.period_optim, epoch + 1, self.lr)

            print("Training complete. Best Loss:", best_loss)

        torch.cuda.empty_cache()

    def fast_compute_pa_f1(self, labels, scores, threshold):
        """
        Fast PA-F1 computation, without any other auxiliary metrics.
        """
        # 1. generate the base predictions
        pred = (scores >= threshold).astype(int)

        # 2. Point Adjustment (PA): if any point within a ground-truth anomaly
        #    segment is predicted as 1, the entire segment counts as correctly detected

        # locate all anomaly segments in the ground truth via 0->1 / 1->0 transitions
        gt_diff = np.diff(np.r_[0, labels, 0])
        starts = np.where(gt_diff == 1)[0]
        ends = np.where(gt_diff == -1)[0]

        # adjust pred for each ground-truth segment
        for s, e in zip(starts, ends):
            if np.sum(pred[s:e]) > 0:  # segment already contains at least one positive prediction
                pred[s:e] = 1  # mark the whole segment as positive

        # 3. compute F1
        # TP: pred=1 & labels=1
        tp = np.sum((pred == 1) & (labels == 1))
        # FP: pred=1 & labels=0
        fp = np.sum((pred == 1) & (labels == 0))
        # FN: pred=0 & labels=1
        fn = np.sum((pred == 0) & (labels == 1))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        return f1

    @torch.no_grad()
    def test(self):
        seed = int(getattr(self, "ckpt_seed", None) or getattr(self, "random_seed", 0))
        tag = str(getattr(self, "ckpt_tag", None) or getattr(self, "exp_tag", "default"))
        w_path = f"weights/{self.dataset}_{self.win_size}_{self.input_size}_{tag}_seed{seed}_best.pt"
        _sd = torch.load(w_path, map_location=self.device)
        _sd = {k: v for k, v in _sd.items()
               if not k.endswith(("total_ops", "total_params"))}
        self.model.load_state_dict(_sd)
        self.model.to(self.device)
        self.model.eval()

        def compute_energy(loader, normalize=False, max_batches=None, warmup=0, profile=False):
            energy = []
            labels = []

            # profiling
            if profile and torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            import time
            t0 = time.time()

            wrap_mode = str(getattr(self, "psp_wrap", "wrap"))
            max_k = int(max(self.periods)) if hasattr(self, "periods") and len(self.periods) > 0 else 0

            for bi, (x_ori, label, idxs) in enumerate(tqdm.tqdm(loader, total=len(loader))):
                if (max_batches is not None) and (max_batches > 0) and (bi >= max_batches):
                    break

                x_ori = x_ori.to(self.device).float()
                label = label.to(self.device)

                # warmup batches: still run forward, but don't record
                is_warm = (profile and bi < warmup)

                base_dataset = loader.dataset.base
                total_len = len(base_dataset)

                if wrap_mode == "drop":
                    valid_bs = []
                    for b in range(x_ori.size(0)):
                        base_idx = int(idxs[b])
                        if (base_idx + max_k) < total_len:
                            valid_bs.append(b)
                    if len(valid_bs) == 0:
                        continue

                    x_ori_use = x_ori[valid_bs]
                    label_use = label[valid_bs]
                    idxs_use = [int(idxs[b]) for b in valid_bs]
                else:
                    x_ori_use = x_ori
                    label_use = label
                    idxs_use = [int(i) for i in idxs]

                loss_fn = nn.MSELoss(reduction='none')
                all_losses = []

                for k in self.periods:
                    rs_list = []
                    for base_idx in idxs_use:
                        if wrap_mode == "wrap":
                            ref_index = (base_idx + k) % total_len
                        else:
                            ref_index = base_idx + k
                        rs_patch, _ = base_dataset[ref_index]
                        rs_list.append(rs_patch)

                    x_ref = torch.tensor(np.stack(rs_list), dtype=torch.float32).to(self.device)

                    recon_k = self.model(x_ori_use, k)
                    loss_map = loss_fn(recon_k, x_ref)  # [B,C,H,W]
                    s_k = self.pixel_pooling(loss_map)  # [B]
                    all_losses.append(s_k.unsqueeze(1))  # [B,1]

                all_losses_tensor = torch.cat(all_losses, dim=1)  # [B,K]
                per_sample_loss = self.expert_pooling(all_losses_tensor)  # [B]

                if not is_warm:
                    energy.extend(per_sample_loss.detach().cpu().numpy())
                    labels.extend(label_use.detach().cpu().numpy())

            if profile and torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.time() - t0

            energy = np.array(energy)
            if normalize and len(energy) > 0:
                energy = (energy - energy.min()) / (energy.max() - energy.min() + 1e-8)

            peak_mem = None
            if profile and torch.cuda.is_available():
                peak_mem = int(torch.cuda.max_memory_allocated())

            return energy, np.array(labels), t, peak_mem

        if int(getattr(self, "profile_cost", 0)) == 1:
            max_batches = int(getattr(self, "profile_batches", 200))
            warmup = int(getattr(self, "profile_warmup", 10))

            print("\n[COST] Profiling online inference (test_loader) ...")
            _, _, t_sec, peak_mem = compute_energy(
                self.test_loader,
                max_batches=max_batches if max_batches > 0 else None,
                warmup=warmup,
                profile=True
            )
            # number of windows processed = len(energy); we didn't return it, but can infer by rerunning a tiny count
            # easiest: use len(_) energy after warmup is excluded; keep by capturing energy
            energy_tmp, _, _, _ = compute_energy(self.test_loader, max_batches=max_batches if max_batches > 0 else None)
            n_win = int(len(energy_tmp))
            thrpt = (n_win / t_sec) if t_sec > 0 else 0.0

            pre = getattr(self, "preproc_stats", None)
            cache_size = None
            cache_dir = getattr(self, "patch_dir", None)
            if pre is not None:
                cache_size = pre.get("cache_size_bytes", None)
                cache_dir = pre.get("cache_dir", cache_dir)

            print("[COST][offline] cache_dir=", cache_dir)
            print("[COST][offline] preprocess_time_s=", None if pre is None else pre.get("preprocess_time_s", None))
            print("[COST][offline] preprocess_peak_rss_bytes=",
                  None if pre is None else pre.get("preprocess_peak_rss_bytes", None))
            print("[COST][offline] cache_size_bytes=", cache_size)

            print(f"[COST][online] profiled_windows={n_win}  time_s={t_sec:.3f}  windows/s={thrpt:.2f}")
            print(f"[COST][online] peak_gpu_mem_bytes={peak_mem}")

            peak_gpu_mb = None if peak_mem is None else (peak_mem / (1024 ** 2))
            cache_size_gb = None if cache_size is None else (cache_size / (1024 ** 3))

            print(f"[COST][TABLE] dataset={self.dataset}")
            print(f"[COST][TABLE] offline_time_s={None if pre is None else pre.get('preprocess_time_s', None)}")
            print(f"[COST][TABLE] cache_size_gb={cache_size_gb}")
            print(f"[COST][TABLE] online_windows_per_s={thrpt:.4f}")
            print(f"[COST][TABLE] peak_gpu_mb={peak_gpu_mb}")

            # save into artifacts/config.json later: attach to cfg dict
            self.cost_profile = {
                "offline": pre,
                "online": {
                    "profile_batches": max_batches,
                    "warmup_batches": warmup,
                    "profiled_windows": n_win,
                    "time_s": float(t_sec),
                    "windows_per_s": float(thrpt),
                    "peak_gpu_mem_bytes": None if peak_mem is None else int(peak_mem),
                }
            }


        print("Getting energy from train set...")
        train_energy, _, _, _ = compute_energy(self.train_loader)

        print("Getting energy from test set...")
        test_energy, test_labels, _, _ = compute_energy(self.test_loader)

        combined_energy = np.concatenate([train_energy, test_energy])
        # anomaly_ratio = float(self.anomaly_ratio)
        # support multi-ratio eval using the same checkpoint
        eval_ratios_str = str(getattr(self, "eval_ratios", "")).strip()
        if eval_ratios_str:
            eval_ratios = [float(x) for x in eval_ratios_str.split(",") if x.strip() != ""]
        else:
            eval_ratios = [float(self.anomaly_ratio)]

        def _eval_one(tag: str, threshold: float, ar: float):
            pred = (test_energy > threshold).astype(int)
            gt = test_labels.astype(int)
            scores_simple = combine_all_evaluation_scores(pred, gt, test_energy)

            print("\n" + "=" * 24)
            print(f"[Eval: {tag}] anomaly_ratio={ar}  threshold={threshold:.6f}")
            for k, v in scores_simple.items():
                print('{0:21} : {1:0.4f}'.format(k, v))
            return {"threshold": float(threshold), **{k: float(v) for k, v in scores_simple.items()}}

        thr_mode = getattr(self, 'thr_mode', 'combined')
        results = {}

        for anomaly_ratio in eval_ratios:
            anomaly_ratio = float(anomaly_ratio)
            run_key_prefix = f"ratio={anomaly_ratio}"

            if thr_mode in ['combined', 'both']:
                thr_combined = np.percentile(combined_energy, 100 - anomaly_ratio)
                results[f'{run_key_prefix}/combined'] = _eval_one(f'combined (train+test) | {run_key_prefix}',
                                                                  thr_combined, anomaly_ratio)

            if thr_mode in ['train_only', 'both']:
                thr_train = np.percentile(train_energy, 100 - anomaly_ratio)
                results[f'{run_key_prefix}/train_only'] = _eval_one(f'train_only | {run_key_prefix}', thr_train, anomaly_ratio)

            if thr_mode == 'calib':
                rng = np.random.RandomState(getattr(self, 'calib_seed', 0))
                n = len(train_energy)
                idx = np.arange(n)
                rng.shuffle(idx)
                n_cal = int(n * getattr(self, 'calib_ratio', 0.2))
                cal_idx = idx[:n_cal]
                calib_energy = train_energy[cal_idx]
                thr_calib = np.percentile(calib_energy, 100 - anomaly_ratio)
                results[f'{run_key_prefix}/calib_train_only'] = _eval_one(f'calib_train_only | {run_key_prefix}',
                                                                          thr_calib)

            # POT: anomaly_ratio is interpreted as a percentage (0.3 -> 0.3%), kept consistent with the original logic
            target_rate = float(anomaly_ratio) / 100.0
            pot_q = float(getattr(self, "pot_q", 0.98))
            pot_min_exc = int(getattr(self, "pot_min_exc", 50))
            if thr_mode in ['pot', 'both_pot']:
                thr_pot = self.pot_threshold(train_energy, target_rate, q=pot_q, min_exc=pot_min_exc)
                results[f'{run_key_prefix}/pot_train_only'] = _eval_one(f'pot_train_only(q={pot_q}) | {run_key_prefix}',
                                                                        thr_pot)

            if thr_mode in ['both_pot']:
                thr_combined = np.percentile(combined_energy, 100 - anomaly_ratio)
                results[f'{run_key_prefix}/combined'] = _eval_one(f'combined (train+test) | {run_key_prefix}',
                                                                  thr_combined)

        # ========= Save artifacts (optional) =========
        if int(getattr(self, 'save_artifacts', 1)) == 1:
            import os, json, time

            # ts = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            seed = int(getattr(self, 'random_seed', 0))
            tag = str(getattr(self, 'exp_tag', 'default'))
            thr_mode = str(getattr(self, 'thr_mode', 'combined'))

            save_dir = os.path.join(
                self.log_dir, 'artifacts', self.data_path,
                f'{tag}_{thr_mode}_seed{seed}'
            )
            os.makedirs(save_dir, exist_ok=True)

            def _to_jsonable(obj):
                import numpy as _np
                if obj is None or isinstance(obj, (bool, int, float, str)):
                    return obj
                if isinstance(obj, (_np.integer, _np.floating)):
                    return obj.item()
                if isinstance(obj, (list, tuple)):
                    return [_to_jsonable(x) for x in obj]
                if isinstance(obj, dict):
                    return {str(k): _to_jsonable(v) for k, v in obj.items()}
                return str(obj)

            # -------- config snapshot --------
            cfg = {
                "dataset": self.dataset,
                "data_path": self.data_path,
                "win_size": int(self.win_size),
                "input_size": int(self.input_size),
                "input_c": int(self.input_c),
                "anomaly_ratio": float(self.anomaly_ratio),
                "thr_mode": thr_mode,
                "exp_tag": tag,
                "periods": [int(p) for p in self.periods],
            }
            # ===== Period sensitivity snapshots =====
            if hasattr(self, "periods_raw"):
                cfg["periods_raw"] = [int(p) for p in self.periods_raw]
            if hasattr(self, "periods_dedup"):
                cfg["periods_dedup"] = [int(p) for p in self.periods_dedup]
            if hasattr(self, "periods_mainharm"):
                cfg["periods_mainharm"] = [int(p) for p in self.periods_mainharm]
            if hasattr(self, "periods_final"):
                cfg["periods_final"] = [int(p) for p in self.periods_final]
            else:
                cfg["periods_final"] = [int(p) for p in self.periods]  # fallback
            cfg["proxy_mode"] = str(getattr(self, "proxy_mode", "var"))
            cfg["proxy_top_m"] = int(getattr(self, "proxy_top_m", 1))
            cfg["proxy_rpca_clip"] = float(getattr(self, "proxy_rpca_clip", 3.0))
            cfg["period_perturb"] = str(getattr(self, "period_perturb", "none"))
            cfg["period_topk"] = int(getattr(self, "period_topk", 5))
            cfg["period_jitter_ratio"] = float(getattr(self, "period_jitter_ratio", 0.05))
            cfg["erf_gamma"] = float(getattr(self, "erf_gamma", 1.6))
            cfg["eval_ratios"] = str(getattr(self, "eval_ratios", ""))
            cfg["period_source"] = str(getattr(self, "period_source", "rwmpd"))
            cfg["k_list_input"] = [int(x) for x in getattr(self, "k_list", [])]
            cfg["periods_after_perturb"] = [int(p) for p in self.periods]
            # proxy_info was already saved in estimate_periods_robust
            if hasattr(self, "period_proxy_info"):
                cfg["period_proxy_info"] = self.period_proxy_info

            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(cfg), f, indent=2)

            # -------- metrics --------
            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump(_to_jsonable(results), f, indent=2)

            # -------- arrays (energies / labels / preds) --------
            arrays = {
                "train_energy": train_energy,
                "test_energy": test_energy,
                "gt": test_labels.astype(int),
            }

            def _safe_key(s: str) -> str:
                return s.replace("/", "__").replace("=", "_")

            for kname, kv in results.items():
                if kname.endswith("/combined"):
                    thr = float(kv["threshold"])
                    arrays[f"pred_{kname.replace('/', '_')}"] = (test_energy > thr).astype(int)
                    arrays[f"thr_{kname.replace('/', '_')}"] = np.array(thr, dtype=np.float64)

            np.savez_compressed(os.path.join(save_dir, "eval_artifacts.npz"), **arrays)

            print(f"\n[Artifacts saved] {save_dir}")