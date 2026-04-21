import argparse
import os
import re
from pathlib import Path
from csv import writer
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 自动检测设备
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

np.seterr('ignore')

# ---------------------- Helpers ----------------------
def log(*a, **k):
    print(*a, flush=True, **k)

VALID_IMG = (".png", ".jpg", ".jpeg")

def is_img(p):
    return p.lower().endswith(VALID_IMG)

def to01(a):
    return a.astype(np.float32) / 255.0

def mse01(a, b):
    a = to01(a); b = to01(b)
    return float(np.mean((a - b) ** 2))

def rmse01(a, b):
    return float(np.sqrt(mse01(a, b)))

def mae01(a, b):
    a = to01(a); b = to01(b)
    return float(np.mean(np.abs(a - b)))

log(f"[metrics] device = {device}")

# ---------------------- CLI ----------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compute metrics for Breast DCE-MRI (Phase 1/2/3)."
    )
    p.add_argument("real_dir", type=str, help="Path to REAL images")
    p.add_argument("fake_dir", type=str, help="Path to FAKE images")
    p.add_argument("--phases", nargs="+", default=["phase1", "phase2", "phase3"],
                   help="List of phase substrings to filter files by.")
    p.add_argument(
        "--metrics", nargs="+",
        default=["psnr", "ssim", "rmse", "mae"], 
        help="Choose from: psnr ms-ssim lpips ssim rmse mae"
    )
    p.add_argument("--limit", type=int, default=None,
                   help="Max number of pairs to evaluate PER PHASE")
    return p.parse_args()

# ---------------------- Pairing Logic (关键修改处) ----------------------
def core_key(name: str) -> str:
    """
    文件名匹配核心逻辑：
    Fake: 19042580_slice_001_fake_phase1 -> 19042580_slice_001_phase1
    Real: 19042580_slice_001_DCE_phase1  -> 19042580_slice_001_phase1
    """
    stem = Path(name).stem.lower()
    
    stem = stem.replace('_real_', '_').replace('_fake_', '_')
    
    return stem

def pair_two_dirs(dir1: str, dir2: str, filter_str: str = None):
    if not os.path.isdir(dir1) or not os.path.isdir(dir2):
        return [], []

    files1 = [f for f in os.listdir(dir1) if is_img(f)]
    files2 = [f for f in os.listdir(dir2) if is_img(f)]
    
    if filter_str:
        files1 = [f for f in files1 if filter_str in f]
        files2 = [f for f in files2 if filter_str in f]

    # 建立映射
    map1 = {core_key(f): f for f in files1}
    map2 = {core_key(f): f for f in files2}

    # 找交集
    common_keys = sorted(set(map1.keys()) & set(map2.keys()))
    
    if not common_keys:
        log(f"[Warning] No matched pairs found for filter '{filter_str}'!")
        # 调试信息：打印几个 key 看看为什么对不上
        if len(files1) > 0: log(f"  Example Key Real: '{core_key(files1[0])}' (Original: {files1[0]})")
        if len(files2) > 0: log(f"  Example Key Fake: '{core_key(files2[0])}' (Original: {files2[0]})")
        return [], []

    f1_paths = [os.path.join(dir1, map1[k]) for k in common_keys]
    f2_paths = [os.path.join(dir2, map2[k]) for k in common_keys]

    return f1_paths, f2_paths

# ---------------------- Metric Functions ----------------------
def load_images(file_path, resize_size=320):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not load image: {file_path}")
        
    if img.shape[0] != resize_size or img.shape[1] != resize_size:
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_LINEAR)
    
    # Expand to 3 channels for uniformity
    img = np.stack([img] * 3, axis=2) 
    return img

def get_metric_function(metric):
    m = metric.lower()
    if m == 'lpips':
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
        fn = LearnedPerceptualImagePatchSimilarity(net_type='alex').to(device)
        def calc(a, b): 
            t_a = torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(device) / 127.5 - 1.0
            t_b = torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(device) / 127.5 - 1.0
            return float(fn(t_a, t_b).item())
        return calc

    elif m == 'ssim':
        from torchmetrics.functional import structural_similarity_index_measure as SSIM
        def calc(a, b):
            t_a = torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            t_b = torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            return float(SSIM(t_a, t_b, data_range=1.0).item())
        return calc

    elif m == 'psnr':
        from torchmetrics import PeakSignalNoiseRatio
        fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        def calc(a, b):
            t_a = torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            t_b = torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(device) / 255.0
            return float(fn(t_a, t_b).item())
        return calc

    elif m == 'rmse':
        return lambda a, b: rmse01(a[:,:,0], b[:,:,0])
    elif m == 'mae':
        return lambda a, b: mae01(a[:,:,0], b[:,:,0])
    else:
        return lambda a, b: 0.0

# ---------------------- Processing Logic ----------------------
def calculate_metrics_for_phase(real_dir, fake_dir, phase_name, metric_list, limit):
    paths1, paths2 = pair_two_dirs(real_dir, fake_dir, filter_str=phase_name)
    
    if len(paths1) == 0:
        return None, 0

    if limit:
        paths1 = paths1[:limit]
        paths2 = paths2[:limit]

    log(f"--- Processing {phase_name}: {len(paths1)} pairs ---")
    
    raw_values = {m: [] for m in metric_list}
    metric_fns = {m: get_metric_function(m) for m in metric_list}

    for i in tqdm(range(len(paths1)), desc=phase_name):
        try:
            img1 = load_images(paths1[i])
            img2 = load_images(paths2[i])

            for m in metric_list:
                val = metric_fns[m](img1, img2)
                raw_values[m].append(val)
        except Exception as e:
            log(f"Error processing pair {i}: {e}")
            continue

    return raw_values, len(paths1)

# ---------------------- Main ----------------------
if __name__ == "__main__":
    args = parse_args()
    
    csv_file = 'metrics_report1.csv'
    
    # 准备一个全局容器，用来装所有 Phase 的所有数据
    global_raw_values = {m: [] for m in args.metrics}
    total_images_count = 0

    # 准备写入 CSV
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        csv_writer = writer(f)
        
        # 写入表头
        if not file_exists:
            headers = ['Timestamp', 'Phase', 'Count'] + \
                      [f"{m}_mean" for m in args.metrics] + \
                      [f"{m}_std" for m in args.metrics]
            csv_writer.writerow(headers)
        
        now_str = str(datetime.now())[:19]

        # 1. 遍历计算每个 Phase
        for phase in args.phases:
            raw_data, count = calculate_metrics_for_phase(
                args.real_dir, args.fake_dir, phase, args.metrics, args.limit
            )
            
            if raw_data:
                # A. 计算当前 Phase 统计数据
                means = []
                stds = []
                for m in args.metrics:
                    vals = raw_data[m]
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    global_raw_values[m].extend(vals)
                
                total_images_count += count
                
                # B. 写入当前 Phase
                row = [now_str, phase, count] + means + stds
                csv_writer.writerow(row)
                log(f"  > {phase} Results Saved (n={count}).")

        # 2. 计算 Combined (All Phases)
        if total_images_count > 0:
            log("\n=== Calculating Combined Average (Phase 1+2+3) ===")
            global_means = []
            global_stds = []
            
            for m in args.metrics:
                all_vals = global_raw_values[m]
                g_mean = np.mean(all_vals)
                g_std = np.std(all_vals)
                
                global_means.append(g_mean)
                global_stds.append(g_std)
                
                log(f"  Overall {m.upper()}: {g_mean:.4f} ± {g_std:.4f}")

            row_global = [now_str, "Combined_Average", total_images_count] + global_means + global_stds
            csv_writer.writerow(row_global)
            log(f"\nDone. Combined results appended to {csv_file}")
        else:
            log("\nNo images processed, skipping global average.")