#!/usr/bin/env python3
"""Paired significance test for two test_model.py evaluation runs.

Assumes both runs used identical worker_id/seed sequences so episodes are
paired element-wise across the two result arrays.

Auto-detects output type:
  - binary (NDE, crash in {0, 1})           -> McNemar exact + Newcombe paired CI
  - weighted (NADE, crash = 0 or weight>0)  -> paired t-test + paired bootstrap CI

Usage:
    # single file per policy
    python training/evaluate.py --orig results/orig/nde_0.npy --new results/new/nde_0.npy

    # directory (concat all *.npy inside) or glob pattern
    python training/evaluate.py --orig results/orig --new results/new
    python training/evaluate.py --orig 'results/orig/nde_*.npy' --new 'results/new/nde_*.npy'
"""
from __future__ import annotations
import argparse
import glob
import math
import os
import sys
import numpy as np

try:
    from scipy.stats import binomtest, norm, ttest_rel
except ImportError:
    print('Requires scipy. Install with: pip install scipy')
    sys.exit(1)

import numpy as np
import os
from scipy.stats import norm
import math

alpha = 0.05
z = norm.isf(q=alpha)

def calculate_val(the_list):
    Mean = []
    Relative_half_width = []
    Var = []
    var_old = 0
    mean_old = 0
    for i in range(len(the_list)):
        if math.isnan(the_list[i]) or math.isinf(the_list[i]):
            the_list[i] = 0.0
        n = i + 1
        mean_new = mean_old + (the_list[i] - mean_old) / n
        Mean.append(mean_new)
        var_new = (n - 1) * var_old / n + (n - 1) * (the_list[i] - mean_old) ** 2 / (n * n)
        Var.append(1.96 * (np.sqrt(var_new / n)))
        Relative_half_width.append(z * (np.sqrt(var_new / n) / (mean_new + 1e-30)))
        var_old = var_new
        mean_old = mean_new
    return Mean, Relative_half_width, Var

def analyze(path):
    crashes = []
    for file in os.listdir(path):
        try:
            data = np.load(os.path.join(path, file), allow_pickle=True).tolist()
            # print([data[i] for i in range(len(data)) if data[i] > 0])
            # if np.max(data) > 1:
            #     print(np.array(data)[np.where(np.array(data) > 1)])
            crashes.extend(data)
        except:
            continue
    # np.save("/home/linxuan/Embodied/go2_mujoco/results/nade_all.npy", np.array(crashes[:200000]))
    mean, rhf, var = calculate_val(crashes)
    print(f'Failure rate: {np.sum(crashes) / len(crashes)}')
    print(f'Mean: {mean[-1]:.6f}, Relative Half Width: {rhf[-1]:.6f}, Variance: {var[-1]:.6f}')
    print(f'Total samples: {len(crashes)}, Num of crashes: {np.sum(np.array(crashes) > 0)}, Max weight: {np.max(crashes)} \n')


def resolve_files(pattern: str) -> list[str]:
    if os.path.isdir(pattern):
        files = sorted(glob.glob(os.path.join(pattern, '*.npy')))
    elif os.path.isfile(pattern):
        files = [pattern]
    else:
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f'no .npy files matched: {pattern}')
    return files


def load_paired(pattern_orig: str, pattern_new: str) -> tuple[np.ndarray, np.ndarray]:
    """Load .npy files from both sides, paired by sorted order.

    - If one side has more files than the other, the extras on the longer side
      are dropped (paired prefix only).
    - Within each file pair, if episode counts differ, both are truncated to
      the shorter length.
    """
    files_o = resolve_files(pattern_orig)
    files_n = resolve_files(pattern_new)
    files_o_base = [os.path.basename(f) for f in files_o]
    files_n_base = [os.path.basename(f) for f in files_n]
    res_files_o = []
    res_files_n = []
    for f in files_o:
        if os.path.basename(f) in files_n_base:
            res_files_o.append(f)
            res_files_n.append(os.path.join(pattern_new, os.path.basename(f)))
    n_pair = min(len(res_files_o), len(res_files_n))

    arrs_o, arrs_n = [], []
    name_o, name_n = [], []
    for fo, fn in zip(res_files_o[:n_pair], res_files_n[:n_pair]):
        ao = np.load(fo, allow_pickle=False).astype(np.float64).reshape(-1)
        an = np.load(fn, allow_pickle=False).astype(np.float64).reshape(-1)
        m = min(len(ao), len(an))
        print(fo, fn)
        print(np.where((ao!=0) & (an==0))[0])
        arrs_o.append(ao[:1])
        arrs_n.append(an[:1])
        name_o.append(fo)
        name_n.append(fn)
        # print(f'    [{os.path.basename(fo):25s} | {os.path.basename(fn):25s}]  n={m}{note}')

    out_o = np.concatenate(arrs_o) if arrs_o else np.zeros(0)
    out_n = np.concatenate(arrs_n) if arrs_n else np.zeros(0)
    print(f'  -> paired total: {len(out_o)} episodes from {n_pair} file pair(s)')
    return out_o, out_n, name_o, name_n


def is_binary(a: np.ndarray, atol: float = 1e-9) -> bool:
    """True if array values are all 0 or 1."""
    if a.size == 0:
        return True
    return bool(np.all((np.abs(a) < atol) | (np.abs(a - 1.0) < atol)))


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI for a proportion, used as building block for Newcombe."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def mcnemar_analysis(orig: np.ndarray, new: np.ndarray, name_o: list, name_n: list) -> None:
    """Paired binary comparison: McNemar exact + Newcombe paired-difference CI."""
    N = len(orig)
    o = orig.astype(bool)
    n = new.astype(bool)
    a = int(np.sum(~o & ~n))
    b = int(np.sum(~o &  n))   # new crashed, orig did not  -> "new worse"
    c = int(np.sum( o & ~n))   # orig crashed, new did not  -> "new better"
    d = int(np.sum( o &  n))

    print(np.array(name_o)[np.where(o & ~n)[0]])
    print(np.array(name_n)[np.where(o & ~n)[0]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--orig', default='training/results',
                    help='Path to .npy file, directory of .npy, or glob pattern for original policy results')
    ap.add_argument('--new', default='training/results2',
                    help='Same as --orig but for the new policy')
    ap.add_argument('--mode', choices=['auto', 'binary', 'weighted'], default='auto',
                    help='Force a test mode instead of auto-detecting from values')
    ap.add_argument('--n_boot', type=int, default=10000,
                    help='Bootstrap iterations (weighted mode only)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f'[load] orig: {args.orig}')
    analyze(args.orig)
    print(f'[load] new : {args.new}')
    analyze(args.new)
    orig, new, name_o, name_n = load_paired(args.orig, args.new)

    N = len(orig)
    print(f'\n[paired] N = {N} episodes')

    mcnemar_analysis(orig, new, name_o, name_n)

if __name__ == '__main__':
    main()
