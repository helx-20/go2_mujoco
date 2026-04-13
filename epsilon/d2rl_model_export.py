import torch
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.tune.registry import register_env
from d2rl_training_env import D2RLTrainingEnv

import yaml, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--yaml_conf', type=str, default='epsilon/d2rl_train.yaml', metavar='N', help='the yaml configuration file path')
args = parser.parse_args()
try:
    with open(args.yaml_conf, 'r') as f:
        yaml_conf = yaml.load(f, Loader=yaml.FullLoader)
except Exception as e:
    print("Yaml configuration file not successfully loaded:", e)


def env_creator(env_config):
    return D2RLTrainingEnv(yaml_conf)
register_env("my_env", env_creator)

ray.init(local_mode=True, include_dashboard=False, ignore_reinit_error=True)
algo = PPO(config={"env": "my_env", "env_config": yaml_conf})

checkpoint_path = "/home/linxuan/ray_results/training_results/PPO_my_env_b851e_00000_0_2026-04-13_12-17-44/checkpoint_000999"  # replace with your checkpoint path
algo.restore(checkpoint_path)

# Try to get the policy and export the model. Newer RLlib policies may provide
# `export_model`. If not, fallback to tracing/saving the underlying torch module.
p = algo.get_policy()
export_dir = "epsilon/model/"
p.export_model(export_dir)

import os
# Load exported model for standalone inference (if present)
loaded_model = torch.load(os.path.join(export_dir, 'model.pt'), map_location=torch.device('cpu'))


def compute_action_torch(observation):
    # Prefer using an exported torch model when available
    if loaded_model is not None:
        obs = torch.tensor(np.expand_dims(np.array(observation, dtype=np.float32), 0))
        try:
            out = loaded_model({"obs": obs})
        except Exception:
            out = loaded_model(obs)
        try:
            action = int(torch.argmax(out[0], dim=-1).item())
        except Exception:
            action = int(np.argmax(np.array(out[0])))
        return action

    # Fallback: use the RLlib algorithm's compute_single_action
    out = algo.compute_single_action(observation, explore=False)
    if isinstance(out, (tuple, list)):
        return out[0]
    return out

def compute_action_torch_continuous(observation):
    # Prefer using an exported torch model when available
    if loaded_model is not None:
        obs = torch.tensor(np.expand_dims(np.array(observation, dtype=np.float32), 0))
        try:
            out = loaded_model({"obs": obs})
        except Exception:
            out = loaded_model(obs)
        try:
            val = float(out[0][0][0])
        except Exception:
            try:
                val = float(np.array(out[0])[0][0])
            except Exception:
                raise RuntimeError("Unexpected model output shape for continuous action")
        action = np.clip((val + 1) * (0.999 - 0.001) / 2 + 0.001, 0.001, 0.999)
        return action

    # Fallback: use the RLlib algorithm's compute_single_action
    out = algo.compute_single_action(observation, explore=False)
    if isinstance(out, (tuple, list)):
        a = out[0]
    else:
        a = out
    # If action is array-like, try to extract scalar
    try:
        return float(a)
    except Exception:
        try:
            return float(np.array(a).item())
        except Exception:
            raise RuntimeError("Unexpected action shape from algo.compute_single_action")

if __name__ == "__main__":
    import gym
    env = D2RLTrainingEnv(yaml_conf)
    for i_episode in range(20):
        reset_ret = env.reset()
        # gymnasium returns (obs, info), gym returns obs
        obs = reset_ret[0] if isinstance(reset_ret, (tuple, list)) else reset_ret
        for t in range(100):
            # Use the algorithm API for computing actions
            ray_action = algo.compute_single_action(obs, explore=False)
            torch_action = compute_action_torch_continuous(obs)
            print(f"Ray action: {ray_action}. Torch action: {torch_action}")
            action = env.action_space.sample()
            step_ret = env.step(action)
            # gymnasium: (obs, reward, terminated, truncated, info)
            if isinstance(step_ret, (tuple, list)) and len(step_ret) == 5:
                obs, reward, terminated, truncated, info = step_ret
                done = bool(terminated or truncated)
            elif isinstance(step_ret, (tuple, list)) and len(step_ret) == 4:
                obs, reward, done, info = step_ret
            else:
                # Fallback: try to unpack generically
                try:
                    obs, reward, done, info = step_ret
                except Exception:
                    raise RuntimeError("Unexpected env.step() return signature: %r" % (step_ret,))

            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()