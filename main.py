import os
import argparse
import numpy as np
import sys
import time
import warnings

from preprocess import preprocess

warnings.filterwarnings('ignore')

from solver import Solver

# ---------------- Determinism ----------------
os.environ["PYTHONHASHSEED"] = "42"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import torch

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("[WARN] deterministic_algorithms:", e)

def _dir_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def _profile_call(func, *args, **kwargs):
    import time
    t0 = time.time()

    # peak RSS (best-effort)
    peak_rss = None
    try:
        import psutil, threading
        proc = psutil.Process(os.getpid())
        peak = 0
        stop = False

        def sampler():
            nonlocal peak, stop
            while not stop:
                try:
                    peak = max(peak, proc.memory_info().rss)
                except Exception:
                    pass
                time.sleep(0.05)

        th = threading.Thread(target=sampler, daemon=True)
        th.start()
        func(*args, **kwargs)
        stop = True
        th.join(timeout=1.0)
        peak_rss = int(peak)
    except Exception:
        # fallback: no psutil
        func(*args, **kwargs)
        try:
            import resource
            # ru_maxrss: KB on Linux, bytes on macOS; you are Ubuntu so KB.
            peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            peak_rss = None

    t = time.time() - t0
    return t, peak_rss

def bytes_to_gb(x):
    return float(x) / (1024 ** 3)

def bytes_to_mb(x):
    return float(x) / (1024 ** 2)

class Logger(object):
    def __init__(self, filename='default.log', add_flag=True, stream=sys.stdout):
        self.terminal = stream
        self.filename = filename
        self.add_flag = add_flag

    def write(self, message):
        mode = 'a+' if self.add_flag else 'w'
        with open(self.filename, mode) as log:
            self.terminal.write(message)
            log.write(message)

    def flush(self):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # ------- model parameters -------
    parser.add_argument('--win_size', type=int, default=64, help='window size of patches')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_heads', type=int, default=1)
    parser.add_argument('--e_layers', type=int, default=3)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--rec_timeseries', action='store_true', default=True)
    parser.add_argument('--head', type=int, default=18)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=True)
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--loss_fuc', type=str, default='margin', choices=['MSE', 'MAE', 'margin'])
    parser.add_argument('--margin', type=float, default=0.05)
    parser.add_argument('--lam', type=float, default=1.0)

    parser.add_argument('--model_name', type=str, default='MPJL')

    # ------- training -------
    parser.add_argument('--index', type=int, default=137)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--input_c', type=int, default=8)
    parser.add_argument('--output_c', type=int, default=8)
    parser.add_argument('--input_size', type=int, default=16)
    parser.add_argument('--dataset', type=str, default='SKAB')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--data_path', type=str, default='SKAB')
    parser.add_argument('--model_save_path', type=str, default='weights')
    parser.add_argument('--anomaly_ratio', type=float, default=0.3)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--random_seed', type=int, default=42)

    # ------- evaluation protocol -------
    parser.add_argument('--thr_mode', type=str, default='combined',
                        choices=['combined', 'train_only'],
                        help='combined: percentile over train+test scores; train_only: percentile over train scores only')
    parser.add_argument('--save_artifacts', type=int, default=1, choices=[0, 1],
                        help='Save eval artifacts (scores, threshold, metrics) under logs/artifacts/')
    parser.add_argument('--exp_tag', type=str, default='default',
                        help='Tag appended to artifact directory and log filenames')
    parser.add_argument('--ckpt_tag', type=str, default=None,
                        help='Load checkpoint saved under this tag instead of --exp_tag')
    parser.add_argument('--ckpt_seed', type=int, default=None,
                        help='Load checkpoint saved with this seed instead of --random_seed')

    # ------- ERF scaling -------
    parser.add_argument('--erf_gamma', type=float, default=1.6,
                        help='ERF scaling factor gamma; controls the pixel footprint assigned to each period expert')

    # ------- period override (hidden) -------
    # Use these only when loading a checkpoint trained with a specific period list.
    # Example: --period_source fixed --k_list 2
    parser.add_argument('--period_source', type=str, default='rwmpd',
                        help=argparse.SUPPRESS)
    parser.add_argument('--k_list', type=str, default=None,
                        help=argparse.SUPPRESS)

    config = parser.parse_args()

    # convert k_list string "2" / "2,4,8" -> list of ints stored on config
    if config.k_list is not None:
        config.k_list = [int(x.strip()) for x in config.k_list.split(',') if x.strip()]

    args = vars(config)

    set_deterministic(config.random_seed)

    # multi-GPU
    config.use_gpu = True if torch.cuda.is_available() and config.use_gpu else False
    if config.use_gpu and config.use_multi_gpu:
        config.devices = config.devices.replace(' ', '')
        device_ids = config.devices.split(',')
        config.device_ids = [int(id_) for id_ in device_ids]
        config.gpu = config.device_ids[0]

    # os.makedirs("result/", exist_ok=True)
    # sys.stdout = Logger("result/" + config.data_path + ".log", sys.stdout)
    os.makedirs(config.log_dir + "/", exist_ok=True)
    # sys.stdout = Logger(config.log_dir + "/" + config.data_path + ".log", sys.stdout)

    # ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_name = f"{config.data_path}_{config.exp_tag}_{config.thr_mode}_seed{config.random_seed}.log"
    sys.stdout = Logger(os.path.join(config.log_dir, log_name), sys.stdout)

    if config.mode == 'train':
        print("\n\n")
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print('================ Hyperparameters ===============')
        for k, v in sorted(args.items()):
            print(f'{k}: {v}')
        print('====================  Train  ===================')

    # preprocess
    base_dir = os.path.dirname(__file__)
    patch_dir = os.path.join(base_dir, './dataset', config.dataset, 'patches',
                             f'win{config.win_size}_img{config.input_size}')
    # if not os.path.exists(patch_dir):
    #     preprocess(config.dataset, './dataset', config.win_size, config.input_size)
    preproc_stats = {
        "cache_dir": patch_dir,
        "cache_exists": os.path.exists(patch_dir),
        "rebuild_cache": int(getattr(config, "rebuild_cache", 0)),
        "preprocess_time_s": None,
        "preprocess_peak_rss_bytes": None,
    }

    if int(getattr(config, "rebuild_cache", 0)) == 1 and os.path.exists(patch_dir):
        import shutil

        print(f"[COST] rebuild_cache=1 -> deleting cache dir: {patch_dir}")
        shutil.rmtree(patch_dir)

    if not os.path.exists(patch_dir):
        print(f"[COST] building TC-GAF cache: {patch_dir}")
        t_s, peak_rss = _profile_call(preprocess, config.dataset, './dataset',
                                      config.win_size, config.input_size)
        preproc_stats["preprocess_time_s"] = float(t_s)
        preproc_stats["preprocess_peak_rss_bytes"] = None if peak_rss is None else int(peak_rss)
        preproc_stats["cache_exists"] = True
    else:
        print(f"[COST] cache exists: {patch_dir} (skip preprocess)")

    # always record disk size (after potential build)
    if os.path.exists(patch_dir):
        preproc_stats["cache_size_bytes"] = int(_dir_size_bytes(patch_dir))
    else:
        preproc_stats["cache_size_bytes"] = 0

    # pass into solver
    config.preproc_stats = preproc_stats
    config.patch_dir = patch_dir

    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path, exist_ok=True)

    solver = Solver(vars(config))
    if config.mode == 'train':
        solver.train()
    solver.test()
