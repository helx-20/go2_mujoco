"""
Process NDE-generated criticality records into per-step dataset files.

Input: folder of nde_*.npy files produced by `tests/nde_test_go2.py` (each file is list of episode dicts)
Output: saved numpy files under `data/processed`:
  - pos.npy (observations with label==1)
  - neg.npy (observations with label==0)

This script flattens episodes into per-timestep samples and splits into train/val/test if requested.
"""
import argparse
import os
import sys
import numpy as np
import tqdm

# ensure project root is on sys.path so 'criticality' can be imported regardless of cwd
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from criticality.utils.data_utils import collect_nde_files, load_criticality_records


def main(args):
    src = args.nde_folder
    out = args.out
    os.makedirs(out, exist_ok=True)

    files = collect_nde_files(src)
    obs_list = []
    label_list = []

    for f in tqdm.tqdm(files):
        recs = load_criticality_records(f)
        for ep in recs:
            ep_obs = ep.get('obs', [])
            ep_actions = ep.get('actions', None)
            ep_label = int(ep.get('label', 0))
            L = len(ep_obs)
            if L == 0:
                continue
            # For crash episodes: only mark last 3 steps as positive, others negative
            if ep_label == 1:
                pos_idx = set(range(max(0, L-3), L))
            else:
                pos_idx = set()

            for t, o in enumerate(ep_obs):
                if t not in pos_idx:
                    if np.random.rand() > 0.02:  # subsample negative steps to reduce class imbalance
                        continue

                o_arr = np.asarray(o, dtype=np.float32)
                # get action if available and same length
                if ep_actions is not None and len(ep_actions) == L:
                    a = np.asarray(ep_actions[t], dtype=np.float32)
                else:
                    a = np.asarray([], dtype=np.float32)

                if a.size > 0:
                    sample = np.concatenate([o_arr.reshape(-1), a.reshape(-1)], axis=0)
                else:
                    sample = o_arr.reshape(-1)

                obs_list.append(sample)
                label_list.append(1 if t in pos_idx else 0)

    if len(obs_list) == 0:
        print('No data found')
        return

    X = np.stack(obs_list, axis=0)
    y = np.asarray(label_list, dtype=np.int64)

    pos = X[y == 1]
    neg = X[y == 0]

    print(f'Collected samples total={len(y)} pos={len(pos)} neg={len(neg)}')

    # stratified split per-class into train/val/test with ratio 8:1:1
    # infer feature dim from available class
    feature_dim = None
    if pos.size > 0:
        feature_dim = pos.shape[1] if pos.ndim > 1 else 1
    elif neg.size > 0:
        feature_dim = neg.shape[1] if neg.ndim > 1 else 1

    def split_and_save(arr, name_prefix, feature_dim=None):
        n = len(arr)

        idx = np.arange(n)
        np.random.shuffle(idx)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]

        train_arr = arr[train_idx]
        val_arr = arr[val_idx]
        test_arr = arr[test_idx]

        np.save(os.path.join(out, f'{name_prefix}_train.npy'), train_arr, allow_pickle=False)
        np.save(os.path.join(out, f'{name_prefix}_val.npy'), val_arr, allow_pickle=False)
        np.save(os.path.join(out, f'{name_prefix}_test.npy'), test_arr, allow_pickle=False)

    split_and_save(pos, 'pos', feature_dim)
    split_and_save(neg, 'neg', feature_dim)

    # also save combined pos.npy/neg.npy for backward compatibility
    np.save(os.path.join(out, 'pos.npy'), pos, allow_pickle=False)
    np.save(os.path.join(out, 'neg.npy'), neg, allow_pickle=False)

    print('Saved processed data in', out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nde_folder', default='/mnt/mnt1/linxuan/go2_data/data/nde', help='folder where nde_*.npy are stored')
    parser.add_argument('--out', default='/mnt/mnt1/linxuan/go2_data/data/stage1', help='output folder for processed arrays')
    args = parser.parse_args()
    main(args)
