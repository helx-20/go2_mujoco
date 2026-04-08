import os
import numpy as np
from typing import List


def collect_nde_files(folder: str) -> List[str]:
	files = []
	for fn in os.listdir(folder):
		if fn.endswith('.npy'):
			files.append(os.path.join(folder, fn))
	return sorted(files)


def load_criticality_records(fn: str):
	try:
		data = np.load(fn, allow_pickle=True)
	except Exception:
		return []
	# data expected to be a list-like of dicts: {'obs': ep_obs, 'actions': ep_actions, 'label': label}
	return list(data)


def flatten_episode_records(records: List[dict]):
	"""
	From list of episode dicts produce two arrays: observations and labels.
	Each episode contains a list of obs (per timestep) and a scalar label.
	We produce one sample per timestep with same episode label.
	"""
	obs_list = []
	labels = []
	for ep in records:
		label = int(ep.get('label', 0))
		ep_obs = ep.get('obs', [])
		for o in ep_obs:
			obs_list.append(np.asarray(o, dtype=np.float32))
			labels.append(label)
	if len(obs_list) == 0:
		return np.zeros((0,)), np.zeros((0,))
	obs_arr = np.stack(obs_list, axis=0)
	labels_arr = np.asarray(labels, dtype=np.int64)
	return obs_arr, labels_arr

