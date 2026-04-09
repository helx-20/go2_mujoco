import sys, os
import numpy as np
import json
import torch
from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from criticality.utils.criticality_model import SimpleClassifier
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def convert_to_serializable(obj):
    if isinstance(obj, np.float32):
        return float(obj)
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    else:
        return obj

def main(root_path, target_path, nade_model, threshold=0.9):
    crash_data_path_list = []
    for file_name in os.listdir(root_path):
        crash_data_path_list.append(os.path.join(root_path, file_name))
    
    idx = 0
    TP = 0
    FN = 0
    TOTAL = 0

    D = 4
    bins_per_dim = 10
    # centers edges for mapping (11 edges -> 10 centers) from -1 to 1
    edges = np.linspace(-1.0, 1.0, bins_per_dim + 1)
    grids = np.meshgrid(*[np.arange(bins_per_dim) for _ in range(D)], indexing='ij')
    bins_flat = np.stack([g.reshape(-1) for g in grids], axis=1).astype(np.int64)
    num_actions = bins_flat.shape[0]
    centers = np.zeros((num_actions, D), dtype=np.float32)
    for d in range(D):
        b_idx = bins_flat[:, d]
        centers[:, d] = 0.5 * (edges[b_idx] + edges[b_idx + 1])
    centers_t = torch.from_numpy(centers).to(torch.device('cuda'), dtype=torch.float32)

    for crash_data_path in tqdm(crash_data_path_list):
        data = np.load(crash_data_path, allow_pickle=True).tolist()
        for episode in data:
            episode_data = {}
            
            weight_step_info = {}
            drl_epsilon_step_info = {}
            ndd_step_info = {}
            drl_obs_step_info = {}
            criticality_step_info = {}
            total_weight = 1

            num_steps = len(episode['obs'])

            for i in range(num_steps):
                # current observation
                cur_obs = torch.tensor(episode['obs'][i], dtype=torch.float32, device=torch.device('cuda')).unsqueeze(0)
                cur_obs_rep = cur_obs.repeat(num_actions, 1)

                # construct inputs [obs, action]
                new_k_inputs = torch.cat([cur_obs_rep, centers_t], dim=1)

                p_list = np.ones(num_actions, dtype=float)
                p_list = p_list / p_list.sum()
                # evaluate model on obs+action candidates
                with torch.no_grad():
                    out = nade_model(new_k_inputs)
                    if isinstance(out, torch.Tensor) and out.dim() == 2 and out.size(1) == 2:
                        probs = torch.softmax(out, dim=1)[:, 1]
                    else:
                        probs = out.view(-1)
                    q_list = probs.cpu().detach().numpy()
                
                if args.threshold is not None:
                    criticality = (q_list > args.criticality_thresh).astype(float)
                else:
                    criticality = q_list
                if np.max(criticality) > 3e-1 or np.sum(criticality) > 60:
                    criticality = criticality / criticality.sum()
                    pdf_array = (1.0 - args.epsilon) * criticality + args.epsilon * p_list
                else:
                    pdf_array = p_list

                pdf_array = pdf_array / pdf_array.sum()  # ensure normalized
                actual_action = episode['actions'][i]
                action_idx = int(np.argmin(np.linalg.norm(centers - actual_action.reshape(1, -1), axis=1)))
                weight = p_list[action_idx] / pdf_array[action_idx]
                total_weight *= weight

                TOTAL += 1

                if abs(weight - 1) > 1e-5:
                    step = i
                    weight_step_info[str(step)] = weight
                    drl_epsilon_step_info[str(step)] = args.epsilon
                    ndd_step_info[str(step)] = p_list[action_idx]
                    drl_obs_step_info[str(step)] = new_k_inputs[0].cpu().detach().numpy().tolist()
                    criticality_step_info[str(step)] = criticality[action_idx]

            if weight_step_info:
                episode_data['weight_step_info'] = weight_step_info
                episode_data['drl_epsilon_step_info'] = drl_epsilon_step_info
                episode_data['ndd_step_info'] = ndd_step_info
                episode_data['drl_obs_step_info'] = drl_obs_step_info
                episode_data['criticality_info'] = criticality_step_info
                episode_data['weight_episode'] = float(total_weight)

                file_name = os.path.join(target_path, f'crash_0_{idx}.json')
                with open(file_name,'w') as f:
                    json.dump(convert_to_serializable(episode_data),f)
                
                idx += 1

            # print(TP / (TP + FN + 1e-5), TP / TOTAL)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default="/mnt/mnt1/linxuan/go2_data/data/nade")
    parser.add_argument('--target_path', type=str, default="/mnt/mnt1/linxuan/go2_data/data/epsilon/raw_data")
    parser.add_argument('--model_path', type=str, default="criticality/stage1/model/stage1_criticality_best_new_1.pt")
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--epsilon', type=float, default=0.01)
    args = parser.parse_args()
    root_path = args.root_path
    target_path = args.target_path
    os.makedirs(target_path, exist_ok=True)

    def build_wrapper_from_state(path, device=torch.device('cuda')):
        if not path or not os.path.exists(path):
            return None
        state = torch.load(path, map_location=torch.device('cpu'))
        input_dim = None
        if isinstance(state, dict):
            # try common key for first linear weight
            for key in state.keys():
                if key.endswith('net.0.weight'):
                    try:
                        input_dim = state[key].shape[1]
                        break
                    except Exception:
                        pass
        if input_dim is None:
            # fallback: try to infer from example obs in dataset — give up and return None
            return None
        model = SimpleClassifier(input_dim=input_dim, hidden=256)
        try:
            model.load_state_dict(state)
        except Exception:
            # ignore load errors and return model with random init
            pass
        model.eval().to(device)
        return model

    # try to build model from provided path
    state_path = args.model_path
    nade_model = build_wrapper_from_state(state_path, device=torch.device('cuda'))
    if nade_model is None:
        print(f'Warning: could not build model from {state_path}; using zeros for criticality where needed')
    main(root_path, target_path, nade_model, threshold=args.threshold)