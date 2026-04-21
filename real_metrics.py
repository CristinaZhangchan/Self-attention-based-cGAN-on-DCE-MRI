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

# ---------------------- Pairing Logic ----------------------
def get_id_slice_key(name: str) -> str:
    """
    提取 ID 和 Slice 部分
    80520748_slice_001_real_phase1.jpg -> 80520748_slice_001
    """
    stem = Path(name).stem.lower()
    # 移除 real/fake 以及 phase 信息，只留 ID 和 Slice
    key = re.sub(r"_(real|fake)?_?phase[0-9]+", "", stem)
    return key

def pair_phases_within_dir(data_dir: str, p_alpha="phase1", p_beta="phase3"):
    if not os.path.isdir(data_dir):
        log(f"[Error] Directory not found: {data_dir}")
        return [], [], []

    all_files = [f for f in os.listdir(data_dir) if is_img(f)]
    
    # 建立映射
    files_p1 = {get_id_slice_key(f): f for f in all_files if p_alpha in f.lower()}
    files_p3 = {get_id_slice_key(f): f for f in all_files if p_beta in f.lower()}

    # 找交集
    common_keys = sorted(set(files_p1.keys()) & set(files_p3.keys()))
    
    p1_paths = [os.path.join(data_dir, files_p1[k]) for k in common_keys]
    p3_paths = [os.path.join(data_dir, files_p3[k]) for k in common_keys]

    return p1_paths, p3_paths, common_keys

# ---------------------- Metric Functions ----------------------
def load_images(file_path, resize_size=320):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if img is None: raise ValueError(f"Could not load: {file_path}")
    if img.shape[0] != resize_size:
        img = cv2.resize(img, (resize_size, resize_size))
    return np.stack([img] * 3, axis=2) 

def get_metric_function(metric):
    m = metric.lower()
    if m == 'ssim':
        from torchmetrics.functional import structural_similarity_index_measure as SSIM
        return lambda a, b: float(SSIM(torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(device)/255.0, 
                                       torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(device)/255.0, 
                                       data_range=1.0).item())
    elif m == 'psnr':
        from torchmetrics import PeakSignalNoiseRatio
        fn = PeakSignalNoiseRatio(data_range=1.0).to(device)
        return lambda a, b: float(fn(torch.from_numpy(a).float().permute(2,0,1).unsqueeze(0).to(device)/255.0, 
                                     torch.from_numpy(b).float().permute(2,0,1).unsqueeze(0).to(device)/255.0).item())
    elif m == 'rmse': return lambda a, b: rmse01(a[:,:,0], b[:,:,0])
    elif m == 'mae': return lambda a, b: mae01(a[:,:,0], b[:,:,0])
    return None

# ---------------------- Main ----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to real images folder")
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "rmse", "mae"])
    args = parser.parse_args()

    # 结果文件名
    detailed_csv = 'motion_analysis_details.csv'
    
    # 1. 查找配对
    p1_list, p3_list, keys = pair_phases_within_dir(args.data_dir, "phase1", "phase3")
    count = len(p1_list)

    if count == 0:
        log("No pairs found. Check your file names.")
        exit()

    log(f"Starting analysis for {count} pairs...")
    metric_fns = {m: get_metric_function(m) for m in args.metrics}

    # 2. 准备 CSV
    headers = ['Slice_ID', 'File_P1', 'File_P3'] + args.metrics
    
    with open(detailed_csv, 'w', newline='') as f:
        csv_writer = writer(f)
        csv_writer.writerow(headers)

        # 3. 逐张计算并立即写入
        all_results = {m: [] for m in args.metrics}
        
        for i in tqdm(range(count)):
            try:
                img1 = load_images(p1_list[i])
                img3 = load_images(p3_list[i])
                
                row_results = []
                for m in args.metrics:
                    val = metric_fns[m](img1, img3)
                    row_results.append(val)
                    all_results[m].append(val)
                
                # 写入这一张图的数据
                csv_writer.writerow([keys[i], Path(p1_list[i]).name, Path(p3_list[i]).name] + row_results)
                
            except Exception as e:
                log(f"Error on {keys[i]}: {e}")

    # 4. 统计离群值参考
    log(f"\nAnalysis Complete. Detailed data saved to {detailed_csv}")
    log("=== Quick Summary (Use these to find outliers) ===")
    for m in args.metrics:
        arr = np.array(all_results[m])
        mean_val = np.mean(arr)
        std_val = np.std(arr)
        # 离群值通常定义为距离平均值 2 倍标准差之外的点
        log(f"{m.upper()}: Mean={mean_val:.4f}, Std={std_val:.4f}")