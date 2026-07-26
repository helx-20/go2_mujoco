import os
import glob
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def _load_sequence(path: str) -> np.ndarray:
	"""Load data following the convention of `evaluate.py`:
	- If directory: merge all `.npy` files (sorted by filename), using `allow_pickle=True`.
	- If file: directly `np.load(..., allow_pickle=True).tolist()`.
	- Otherwise treat as glob pattern using `glob.glob` and load matched files sorted.
	Returns a 1D numpy array (np.asarray).
	"""
	def load_npy_file(fpath: str):
		try:
			data = np.load(fpath, allow_pickle=True)
			# If numpy array, try converting to list
			try:
				return data.tolist()
			except Exception:
				return np.asarray(data).ravel().tolist()
		except Exception:
			# If not .npy (or corrupted), try loadtxt
			try:
				return np.loadtxt(fpath, delimiter=',').ravel().tolist()
			except Exception:
				return []

	items = []
	if os.path.isdir(path):
		files = sorted(glob.glob(os.path.join(path, '*.npy')))
		for f in files:
			items.extend(load_npy_file(f))
		return np.asarray(items)

	if os.path.isfile(path):
		items = load_npy_file(path)
		return np.asarray(items)

	# treat as glob pattern
	matches = sorted(glob.glob(path))
	if matches:
		for f in matches:
			if os.path.isdir(f):
				# recursively load directory
				items.extend(_load_sequence(f).tolist())
			else:
				items.extend(load_npy_file(f))
		return np.asarray(items)

	# try common extensions
	for ext in ('.npy',):
		if os.path.exists(path + ext):
			return _load_sequence(path + ext)

	raise FileNotFoundError(f"Cannot find data for path: {path}")


def plot_failure_rates(paths: List[str], labels: Optional[List[str]] = None, out_path: str = 'failure_rates.png') -> None:
	"""Plot cumulative failure rate line chart with 95% CI shaded half-width for each path.

	paths: list, each item a file or directory (see `_load_sequence` loading rules).
	labels: optional list of curve labels.
	out_path: output image path.
	"""
	z = norm.ppf(1 - 0.05 / 2)  # 95% CI

	plt.figure(figsize=(10, 6), dpi=120)

	if labels is None:
		labels = [os.path.basename(p) or p for p in paths]

	colors = plt.get_cmap('tab10')
	# Red color consistent with tests/draw_RHF.py
	primary_red = '#8B0000'
	primary_fill = (1.0, 215/255, 215/255)

	# Per user requirement: x-axis is `paths` index (iteration), each path aggregated to one point (merge all .npy)
	means = []
	lowers = []
	uppers = []
	xs = list(range(len(paths)))

	for idx, path in enumerate(paths):
		try:
			data = _load_sequence(path)
		except FileNotFoundError:
			print(f"Warning: data not found for {path}, skipped.")
			means.append(np.nan)
			lowers.append(np.nan)
			uppers.append(np.nan)
			continue

		arr = np.asarray(data).ravel()
		if arr.size == 0:
			print(f"Warning: empty data for {path}, skipped.")
			means.append(np.nan)
			lowers.append(np.nan)
			uppers.append(np.nan)
			continue

		N = arr.size
		mean = float(np.nanmean(arr))
		# Standard error: use sample std / sqrt(N) for both binary and weighted data
		if N > 1:
			se = float(np.nanstd(arr, ddof=1) / np.sqrt(N))
		else:
			se = 0.0
		half = float(z * se)

		lower = mean - half
		upper = mean + half
		# Clip probability values (if data is probability/failure rate)
		lower = max(lower, 0.0)
		upper = min(upper, 1.0)

		means.append(mean)
		lowers.append(lower)
		uppers.append(upper)

	means = np.array(means)
	lowers = np.array(lowers)
	uppers = np.array(uppers)

	# Draw lines and confidence interval bands
	for idx in range(len(paths)):
		if idx == 0:
			line_color = primary_red
			fill_color = primary_fill
			alpha = 0.5
		else:
			line_color = colors(idx % 10)
			fill_color = line_color
			alpha = 0.25

	plt.plot(xs, means, '-o', color=primary_red, linewidth=2)
	#plt.fill_between(xs, lowers, uppers, color=primary_fill, alpha=0.5)
	# Labels show value at each point
	for xi, m in zip(xs, means):
		if not np.isnan(m):
			plt.text(xi, m, f"{m:.2e}", ha='center', va='bottom', fontsize=14)

	plt.xlabel('Iteration', fontsize=16)
	plt.ylabel('Failure rate', fontsize=16)
	plt.legend(fontsize=14)
	plt.grid(True, alpha=0.3)
	plt.ylim(1e-4, 4e-4)
	plt.xticks(fontsize=14)
	plt.yticks(fontsize=14)

	out_dir = os.path.dirname(out_path)
	if out_dir and not os.path.exists(out_dir):
		os.makedirs(out_dir, exist_ok=True)

	plt.tight_layout()
	plt.savefig(out_path, dpi=300)
	plt.close('all')


if __name__ == '__main__':
	# Take result list as input, output to results/failure_rates.png under workspace
	results_list = ['training/results_all/results_origin', 'training/results_all/results_round1', 'training/results_all/results_round2', 'training/results_all/results_round3', 'training/results_all/results_round4', 'training/results_all/results_round5', 'training/results_all/results_round6', 'training/results_all/results_round6_thresh05']
	plot_failure_rates(results_list, out_path='training/failure_rates.png')
	print('Saved failure plot to training/failure_rates.png')