#!/usr/bin/env python3
"""Plot training logs (episode length, episode reward, loss curves).

This script plots episode length, episode reward, and loss curves from
explicit CSV files produced by SB3 (`progress.csv`) or Gym `Monitor`.

Usage examples:
    python training/plot_training.py --progress-csv training/logs/run/progress.csv --out out.png --smooth 20
    python training/plot_training.py --csv monitor.csv --out out.png
"""
from __future__ import annotations
import argparse
import os
import glob
import sys
from typing import Optional, Tuple, List

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
except Exception as e:
    print('Missing dependency:', e)
    print('Please install: pip install pandas matplotlib numpy')
    sys.exit(1)


# note: automatic discovery removed; require explicit --csv or --progress-csv


def detect_monitor_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c.lower() for c in df.columns]
    reward_col = None
    len_col = None
    for c in df.columns:
        cl = c.lower()
        if cl in ("r", "reward", "episode_reward", "ep_rew", "rew"):
            reward_col = c
        if cl in ("l", "l_", "length", "episode_length", "ep_len"):
            len_col = c
    # try common pair r,l
    if reward_col is None and "r" in cols:
        reward_col = df.columns[cols.index("r")]
    if len_col is None and "l" in cols:
        len_col = df.columns[cols.index("l")]
    return reward_col, len_col


def find_loss_columns(df: pd.DataFrame) -> List[str]:
    losses = [c for c in df.columns if "loss" in c.lower() or "policy/" in c.lower() or "value_loss" in c.lower()]
    # also include plain 'loss' if present
    if not losses and "loss" in [c.lower() for c in df.columns]:
        losses = [c for c in df.columns if c.lower() == "loss"]
    return losses


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    # use 'same' so output length matches input and alignment is preserved
    return np.convolve(x, np.ones(window) / window)


def moving_std(x: np.ndarray, window: int) -> np.ndarray:
    """Estimate moving standard deviation with same alignment as moving_average."""
    x = np.asarray(x)
    if window <= 1 or x.size == 0:
        return np.zeros_like(x)
    sq = x * x
    kernel = np.ones(window) / window
    m = np.convolve(x, kernel)
    m2 = np.convolve(sq, kernel)
    var = m2 - m * m
    var[var < 0] = 0.0
    return np.sqrt(var)


def clean_xy(x_raw: np.ndarray, y_raw: np.ndarray, trim_last: int = 0) -> tuple:
    """Remove non-finite entries from x_raw/y_raw and optionally trim last N points.

    Returns (x, y) as 1-D numpy arrays with finite values only.
    """
    x_raw = np.asarray(x_raw).reshape(-1)
    y_raw = np.asarray(y_raw).reshape(-1)
    mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
    if mask.sum() == 0:
        return np.array([]), np.array([])
    x = x_raw[mask]
    y = y_raw[mask]
    if trim_last and y.size > trim_last:
        x = x[:-trim_last]
        y = y[:-trim_last]

    return x, y


def plot_logs(progress_csv: Optional[str], out: str, smooth: int = 1, show: bool = False, trim_last: int = 0):
    plotted_any = False

    # Pre-read progress CSV to detect loss columns so we can allocate subplots
    dfp = None
    losses = []
    if progress_csv and os.path.isfile(progress_csv):
        dfp = pd.read_csv(progress_csv, comment='#')
        losses = find_loss_columns(dfp)

    n_losses = max(1, len(losses))
    total_plots = 2 + n_losses
    cols = 2
    rows = (total_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10 * cols / 2, 3 * rows), tight_layout=True)
    axes = np.asarray(axes).reshape(-1)
    # hide any extra axes
    for i in range(total_plots, axes.size):
        axes[i].set_visible(False)
    ax_r = axes[0]
    ax_l = axes[1]
    ax_loss_axes = axes[2:2 + n_losses]

    if dfp is not None:
        # plot rollout ep reward / ep len from progress CSV if present
        for col in ("rollout/ep_rew_mean", "rollout/ep_len_mean", "ep_rew_mean", "ep_len_mean"):
            if col in dfp.columns:
                y_raw = pd.to_numeric(dfp[col], errors='coerce').to_numpy()
                # build x_raw first so we can filter rows where x or y is NaN
                if 'time/total_timesteps' in dfp.columns:
                    x_raw = pd.to_numeric(dfp['time/total_timesteps'], errors='coerce').to_numpy()
                elif 'rollout/ep_len_mean' in dfp.columns:
                    ep_len = pd.to_numeric(dfp['rollout/ep_len_mean'], errors='coerce').to_numpy()
                    ep_len = np.nan_to_num(ep_len, nan=0.0)
                    x_raw = np.cumsum(ep_len)
                else:
                    x_raw = np.arange(1, len(y_raw) + 1)

                # clean x/y from NaN/Inf and apply trim-last before smoothing
                x, y = clean_xy(x_raw, y_raw, trim_last=trim_last)
                # smooth only the y-values; keep x (timesteps) monotonic
                y_s = moving_average(y, smooth)
                x_s = x
                # align lengths: moving_average; trim x_s to y_s length if needed
                if x_s.size != y_s.size:
                    # resample/trim to match
                    minlen = min(x_s.size, y_s.size)
                    x_s = x_s[:minlen]
                    y_s = y_s[:minlen]
                x_s = x_s - np.min(x_s) if x_s.size else x_s
                std_s = moving_std(y, smooth)
                std_s = std_s[:y_s.size]
                if 'rew' in col:
                    line, = ax_r.plot(x_s, y_s, label=f'{os.path.basename(progress_csv)}:{col}')
                    ax_r.fill_between(x_s, y_s - std_s, y_s + std_s, color=line.get_color(), alpha=0.2)
                    ax_r.set_title('Episode Reward')
                    ax_r.set_xlabel('timesteps')
                    ax_r.set_ylabel('reward')
                    plotted_any = True
                else:
                    line, = ax_l.plot(x_s, y_s, label=f'{os.path.basename(progress_csv)}:{col}')
                    ax_l.fill_between(x_s, y_s - std_s, y_s + std_s, color=line.get_color(), alpha=0.2)
                    ax_l.set_title('Episode Length')
                    ax_l.set_xlabel('timesteps')
                    ax_l.set_ylabel('length')
                    plotted_any = True

        # plot each loss in its own subplot for clarity
        if losses:
            for idx, c in enumerate(losses):
                y = pd.to_numeric(dfp[c], errors='coerce').to_numpy()
                y = y[~np.isnan(y)]
                if y.size == 0:
                    continue
                y_raw = pd.to_numeric(dfp[c], errors='coerce').to_numpy()
                mask = ~np.isnan(y_raw)
                if mask.sum() == 0:
                    continue
                # build x_raw first so we can filter invalid rows
                if 'time/total_timesteps' in dfp.columns:
                    x_raw = pd.to_numeric(dfp['time/total_timesteps'], errors='coerce').to_numpy()
                elif 'rollout/ep_len_mean' in dfp.columns:
                    ep_len = pd.to_numeric(dfp['rollout/ep_len_mean'], errors='coerce').to_numpy()
                    ep_len = np.nan_to_num(ep_len, nan=0.0)
                    x_raw = np.cumsum(ep_len)
                else:
                    x_raw = np.arange(1, len(y_raw) + 1)

                # clean x/y from NaN/Inf and apply trim-last before smoothing
                x, y = clean_xy(x_raw, y_raw, trim_last=trim_last)
                if x.size == 0:
                    continue

                # smooth only the y-values; keep x (timesteps) monotonic
                y_s = moving_average(y, smooth)
                std_s = moving_std(y, smooth)
                x_s = x
                if x_s.size != y_s.size:
                    minlen = min(x_s.size, y_s.size)
                    x_s = x_s[:minlen]
                    y_s = y_s[:minlen]
                x_s = x_s - np.min(x_s) if x_s.size else x_s
                std_s = std_s[:minlen]
                ax = ax_loss_axes[min(idx, len(ax_loss_axes) - 1)]
                line, = ax.plot(x_s, y_s, label=c)
                ax.fill_between(x_s, y_s - std_s, y_s + std_s, color=line.get_color(), alpha=0.2)
                ax.set_title(c)
                ax.set_xlabel('timesteps')
                ax.set_ylabel('loss')
                plotted_any = True

    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='best')

    # Use scientific notation on x-axis for all visible axes
    for ax in axes:
        if not getattr(ax, 'get_visible', lambda: True)():
            continue
        try:
            ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        except Exception:
            pass

    if not plotted_any:
        print('No suitable data found to plot. Checked:', progress_csv)

    plt.savefig(out, dpi=600)
    print('Saved plot to', out)
    if show:
        plt.show()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', type=str, default='training/models/run_nade2/logs/run_nade2')
    p.add_argument('--progress-csv', type=str, default='progress.csv', help='Direct path to a progress CSV (preferred)')
    p.add_argument('--out', type=str, default='training_plot.png', help='Output image path')
    p.add_argument('--smooth', type=int, default=10, help='Moving-average window for smoothing')
    p.add_argument('--trim-last', type=int, default=0, help='Number of points to trim from end of each series')
    p.add_argument('--show', action='store_true', help='Show plot interactively')
    args = p.parse_args()
    args.progress_csv = os.path.join(args.dir, args.progress_csv)
    args.out = os.path.join(args.dir, args.out)

    # Respect explicit arguments: if user passed --csv or --progress-csv, prefer them
    progress_csv = args.progress_csv if args.progress_csv else None

    if progress_csv is None:
        print('Please provide either --progress-csv (see --help).')
        sys.exit(2)

    plot_logs(progress_csv, args.out, smooth=max(1, args.smooth), show=args.show, trim_last=int(args.trim_last))


if __name__ == '__main__':
    main()
