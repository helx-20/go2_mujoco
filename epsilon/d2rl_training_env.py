from gymnasium import spaces, core
import os, glob
import random
import json
import numpy as np
import logging
import yaml


class D2RLTrainingEnv(core.Env):
    def __init__(self, yaml_conf):
        self.yaml_conf = yaml_conf
        self.action_space = spaces.Box(low=0.001, high=0.999, shape=(1,))
        self.observation_space = spaces.Box(low=-10, high=10, shape=(56,))

        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0  # some customized metric logging
        self.total_episode, self.total_steps = 0, 0
        
        data_folder = os.path.join(yaml_conf['local_dir'], yaml_conf['data_folder'])
        self.crash_data_path_list, self.crash_data_weight_list = self.get_path_list(data_folder)
        self.all_data_path_list = self.crash_data_path_list
        self.episode_data_path = ""
        self.episode_data = None

        self.unwrapped.trials = 100
        self.unwrapped.reward_threshold = 1.5

    def get_path_list(self, data_folder):
        if os.path.exists(os.path.join(data_folder, "crash_weight_dict.json")):
            with open(os.path.join(data_folder, "crash_weight_dict.json")) as data_file:
                crash_weight_dict = json.load(data_file)
                crash_data_path_list = list(crash_weight_dict.keys())
                crash_data_weight_list = [crash_weight_dict[path] for path in crash_data_path_list]
        else:
            raise ValueError("No weight information!")
        return crash_data_path_list, crash_data_weight_list

    def reset(self, episode_data_path=None, *, seed=None, options=None):
        """
        Gymnasium-style reset: signature accepts (seed, options) and returns (obs, info).
        We keep backward-compatible behavior by delegating to _reset and returning an empty info dict.
        """
        self.constant, self.weight_reward, self.exposure, self.positive_weight_reward = 0, 0, 0, 0
        self.total_episode = 0
        self.total_steps = 0
        self.episode_data_path = ""
        self.episode_data = None
        obs = self._reset(episode_data_path)
        return obs, {}

    def filter_episode_data(self, episode_data):
        invalid_timestep_list = []
        for timestep in episode_data["weight_step_info"]:
            if abs(episode_data["weight_step_info"][timestep] - 1.0) < 1e-5:
                invalid_timestep_list.append(timestep)
                logging.debug(f"popping out {episode_data['weight_step_info']}")
        for invalid_time_step in invalid_timestep_list:
            episode_data["weight_step_info"].pop(invalid_time_step, None)
            episode_data["drl_epsilon_step_info"].pop(invalid_time_step, None)
            episode_data["ndd_step_info"].pop(invalid_time_step, None)
            episode_data["drl_obs_step_info"].pop(invalid_time_step, None)
        return episode_data

    def sample_data_this_episode(self):
        if self.crash_data_weight_list:
            episode_data_path = random.choices(self.crash_data_path_list, weights=self.crash_data_weight_list)[0]
        else:
            raise ValueError("No weight information!")
        return episode_data_path

    def _reset(self, episode_data_path=None):
        self.total_episode += 1
        if not episode_data_path:
            self.episode_data_path = self.sample_data_this_episode()
        else:
            self.episode_data_path = episode_data_path

        #print('reset--episode data path:',self.episode_data_path)

        self.episode_data_path =  self.episode_data_path
        with open(self.episode_data_path) as data_file:
            self.episode_data = self.filter_episode_data(json.load(data_file))
        if self.episode_data is not None:
            all_obs = self.episode_data["drl_obs_step_info"]
            time_step_list = list(all_obs.keys())
            if len(time_step_list):
                init_obs = np.array(all_obs[time_step_list[0]], dtype=np.float32)
                return init_obs
            else:
                return self._reset()
        else:
            return self._reset()

    def step(self, action):
        """
        Gymnasium-style step: return (obs, reward, terminated, truncated, info).
        We treat `truncated` as False unless a horizon-specific truncation is implemented.
        """
        # support actions passed as numpy scalars or tensors
        # accept scalar, numpy array, list, or torch tensor
        try:
            # convert torch tensors to numpy
            if hasattr(action, 'cpu'):
                action = action.cpu().numpy()
        except Exception:
            pass
        obs = self._get_observation()
        terminated = self._get_done()
        time_step_list = list(self.episode_data["drl_obs_step_info"].keys())
        # store action as a python list for JSON-serializability
        stored_action = action.item()
        self.episode_data["drl_epsilon_step_info"][time_step_list[self.total_steps]] = stored_action
        reward = self._get_reward()
        info = self._get_info()
        self.total_steps += 1
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_info(self):
        return {}

    def close(self):
        return

    def _get_observation(self):
        all_obs = self.episode_data["drl_obs_step_info"]
        time_step_list = list(all_obs.keys())
        obs = np.float32(all_obs[time_step_list[self.total_steps]])
        return obs

    def _get_reward(self):  # ! Aim to remove the magnitude of the environment
        stop = self._get_done()
        if not stop:
            return 0
        else:
            drl_epsilon_weight = self._get_drl_epsilon_weight(self.episode_data["weight_step_info"],
                                                              self.episode_data["drl_epsilon_step_info"],
                                                              self.episode_data["ndd_step_info"],
                                                              self.episode_data["criticality_info"])
            clip_reward_threshold = self.yaml_conf["clip_reward_threshold"]
            q_amplifier_reward = clip_reward_threshold - drl_epsilon_weight * 500 * clip_reward_threshold  # drl epsilon weight reward
            if q_amplifier_reward < -clip_reward_threshold:
                q_amplifier_reward = -clip_reward_threshold
            print("final_reward:", q_amplifier_reward)

            return q_amplifier_reward

    def _get_drl_epsilon_weight(self, weight_info, epsilon_info, ndd_info, criticality_info):
        total_q_amplifier = 1
        for timestep in epsilon_info:
            if timestep in weight_info:
                if weight_info[timestep] > 1:
                    total_q_amplifier = total_q_amplifier * (1 / (epsilon_info[timestep]))
                elif weight_info[timestep] < 0.999:
                    ndd_tmp = ndd_info[timestep]
                    criticality_tmp = criticality_info[timestep]
                    # total_q_amplifier = total_q_amplifier * (ndd_tmp / (epsilon_info[timestep] * ndd_tmp + (1 - epsilon_info[timestep]) * criticality_tmp + ndd_tmp)) # * ndd_tmp
                    total_q_amplifier = total_q_amplifier * (1 / (1 - epsilon_info[timestep])) * ndd_tmp
        print("mean epsilon:", np.mean([epsilon_info[timestep] for timestep in epsilon_info]))
        return total_q_amplifier

    def _get_done(self):
        stop = False
        if self.total_steps == len(self.episode_data["drl_obs_step_info"].keys()) - 1:
            stop = True
        return stop


if __name__ == "__main__":
    f = open('d2rl_train.yaml', 'r', encoding='utf-8')
    yaml_conf = yaml.safe_load(f)

    env = D2RLTrainingEnv(yaml_conf)

    for i in range(100):
        print(f'episode {i},-------------------')
        obs = env.reset()
        while True:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            if done:
                break