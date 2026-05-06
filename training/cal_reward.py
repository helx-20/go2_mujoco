import numpy as np
import os

data_dir = '/mnt/mnt1/linxuan/go2_data/data/training/round2'

avg_rews = []
avg_crash_rate = []
avg_useful_rews = []
avg_useful_crash_rate = []
avg_useful_crash_rate_no_weight = []
total_episodes = 0
for filename in os.listdir(data_dir):
    if filename.endswith('.npy') and not filename.startswith('all'):
        path = os.path.join(data_dir, filename)
        try:
            data = np.load(path, allow_pickle=True).item()
        except Exception as e:
            continue
        total_episodes += 100
        rews = np.array(data['rewards'], dtype=np.float32)
        dones = np.array(data['dones'], dtype=bool)
        weights = np.array(data['weights'], dtype=np.float32)
        useful = np.array(data['useful'], dtype=bool)
        total_weight = 1.0
        cur_useful = False
        for idx in range(len(weights)):
            if useful[idx]:
                cur_useful = True
            if weights[idx] != weights[idx-1] and weights[idx] > 0:
                total_weight *= weights[idx]
            if dones[idx]:
                # if rews[idx] < 0 and total_weight > 0.5:
                #     total_weight = 0.0
                useful[idx] = cur_useful
                weights[idx] = total_weight
                total_weight = 1.0
                cur_useful = False
        
        all_idx = np.where(dones == True)[0]
        useful_idx = np.where((dones == True) & (useful == True))[0]
        avg_rews.extend((rews[all_idx] * weights[all_idx]).tolist())
        avg_crash_rate.extend(((rews[all_idx] < 0) * weights[all_idx]).tolist())
        avg_useful_rews.extend((rews[useful_idx] * weights[useful_idx]).tolist())
        avg_useful_crash_rate.extend(((rews[useful_idx] < 0) * weights[useful_idx]).tolist())
        avg_useful_crash_rate_no_weight.extend(((rews[useful_idx] < 0) * 1.0).tolist())

    print(f'Average reward: {np.sum(avg_rews)/total_episodes:.6f}, crash_rate: {np.sum(avg_crash_rate)/total_episodes:.6f}')
    print(f'Average useful reward: {np.sum(avg_useful_rews)/total_episodes:.6f}, useful crash_rate: {np.sum(avg_useful_crash_rate)/total_episodes:.6f}, ratio: {np.sum(avg_useful_crash_rate) / (np.sum(avg_crash_rate) + 1e-30):.6f}, useful crash_rate no weight: {np.sum(avg_useful_crash_rate_no_weight)/total_episodes:.6f}')
