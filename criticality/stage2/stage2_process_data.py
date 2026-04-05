"""
Build offline replay buffer for DQN from NDE criticality episode records.

Rules:
 - Use the per-step labels: crash episodes have last 1 steps labeled 1, others 0.
 - Reward for a transition is 1.0 if the step's label==1, else 0.0.

Output: `replay_buffer.npy` under provided `--out` folder.
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
    replay_buffer_pos = []
    replay_buffer_neg = []
    for f in tqdm.tqdm(files):
        recs = load_criticality_records(f)
        for ep in recs:
            ep_obs = ep.get('obs', [])
            ep_actions = ep.get('actions', [])
            ep_label = int(ep.get('label', 0))
            L = len(ep_obs)
            if L == 0:
                continue
            if ep_label == 1:
                pos_idx = set(range(max(0, L-1), L))
            else:
                pos_idx = set()

            for t in range(L):
                cur = ep_obs[t]
                next_o = ep_obs[t+1] if t+1 < L else cur
                action = ep_actions[t] if t < len(ep_actions) else np.array([], dtype=np.float32)
                done = 1.0 if (t+1 >= L) else 0.0
                reward = 1.0 if ep_label == 1 else 0.0
                if ep_label == 0:
                    if np.random.rand() > 0.01:  # subsample negative transitions to reduce class imbalance
                        continue
                    else:
                        replay_buffer_neg.append({
                            'input': np.asarray(cur, dtype=np.float32),
                            'next_input': np.asarray(next_o, dtype=np.float32),
                            'action': np.asarray(action, dtype=np.float32),
                            'reward': reward,
                            'done': done
                        })
                else:
                    replay_buffer_pos.append({
                        'input': np.asarray(cur, dtype=np.float32),
                        'next_input': np.asarray(next_o, dtype=np.float32),
                        'action': np.asarray(action, dtype=np.float32),
                        'reward': reward,
                        'done': done
                    })

    if len(replay_buffer_pos) == 0 and len(replay_buffer_neg) == 0:
        print('No transitions collected')
        return

    np.save(os.path.join(out, 'replay_buffer_pos.npy'), np.array(replay_buffer_pos, dtype=object), allow_pickle=True)
    np.save(os.path.join(out, 'replay_buffer_neg.npy'), np.array(replay_buffer_neg, dtype=object), allow_pickle=True)
    print(f'Saved positive replay buffer with {len(replay_buffer_pos)} transitions to', os.path.join(out, 'replay_buffer_pos.npy'))
    print(f'Saved negative replay buffer with {len(replay_buffer_neg)} transitions to', os.path.join(out, 'replay_buffer_neg.npy'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--nde_folder', default='/mnt/mnt1/linxuan/go2_data/data/nde', help='folder where nde_*.npy are stored')
    parser.add_argument('--out', default='/mnt/mnt1/linxuan/go2_data/data/stage2', help='output folder for processed arrays')
    args = parser.parse_args()
    main(args)
