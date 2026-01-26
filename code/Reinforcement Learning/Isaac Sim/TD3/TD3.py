'''
./python3 tensorboard --logdir=/home/ym/.local/share/ov/pkg/isaac-sim-4.0.0/td3_logs/tensorboard
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from torch import nn
import torch
from DDPG_env import MazeEnv
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym

class CustomCallback(BaseCallback):
    def __init__(self, return_file, model_path, check_freq, log_dir, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.return_file = return_file
        self.model_path = model_path
        self.check_freq = check_freq
        self.episode_count = 0
        self.return_list = []
        self.current_return = 0
        self.writer = SummaryWriter(log_dir)
        if os.path.exists(return_file):
            with open(return_file, 'r') as f:
                self.return_list = json.load(f)

    def _on_step(self) -> bool:
        self.current_return += self.locals['rewards'][0]

        if self.locals['dones'][0]:
            self.episode_count += 1
            self.return_list.append(float(self.current_return))
            print(f"Episode {self.episode_count}: Return {self.current_return}")

            self.writer.add_scalar('Return', self.current_return, self.episode_count)
            
            self.current_return = 0

            if self.episode_count % self.check_freq == 0:
                self.plot_returns()
                save_path = f"{self.model_path}_ep{self.episode_count}.zip"
                self.model.save(save_path)
                print(f"Model saved at {save_path}")

        return True

    def plot_returns(self):
        with open(self.return_file, 'w') as f:
            json.dump(self.return_list, f)

        plt.plot(self.return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('TD3 on MazeEnv')
        plt.savefig(self.return_file.replace('.json', '.png'))
        plt.close()

def train():
    log_dir = "td3_logs/"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return_file = os.path.join(log_dir, "returns.json")
    model_path = os.path.join(model_dir, "latest_model")
    tensorboard_log = os.path.join(log_dir, "tensorboard")

    env = make_vec_env(lambda: MazeEnv(headless=False), n_envs=1)

    td3_params = {
        'learning_rate': 1e-3,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': (1, 'step'),
        'gradient_steps': 1,
        'action_noise': None,
        'policy_delay': 2,
        'target_policy_noise': 0.2,
        'target_noise_clip': 0.5,
        'tensorboard_log': tensorboard_log,
        'policy_kwargs': {
            'net_arch': [256, 256],
            'activation_fn': nn.ReLU,
        },
        'verbose': 1,
        'device': 'cuda'
    }

    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip with new parameters")
        new_model = TD3('MlpPolicy', env, **td3_params)
        old_model = TD3.load(model_path + ".zip", env=env)
        new_model.policy.load_state_dict(old_model.policy.state_dict())
    else:
        print("No existing model found, training a new model.")
        new_model = TD3('MlpPolicy', env, **td3_params)

    callback = CustomCallback(return_file, model_path, check_freq=10, log_dir=tensorboard_log)

    new_model.learn(total_timesteps=1e7, callback=callback)

if __name__ == "__main__":
    train()
