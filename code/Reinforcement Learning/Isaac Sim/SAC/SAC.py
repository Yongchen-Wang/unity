'''
./python3 tensorboard --logdir=/home/ym/.local/share/ov/pkg/isaac-sim-4.0.0/sac_logs/tensorboard
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import FlattenExtractor
from torch import nn
import torch
from env import MazeEnv
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
        # 更新当前回合的回报
        self.current_return += self.locals['rewards'][0]

        # 检查是否到达回合结束
        if self.locals['dones'][0]:
            self.episode_count += 1
            self.return_list.append(float(self.current_return))
            print(f"Episode {self.episode_count}: Return {self.current_return}")

            # 记录到 TensorBoard
            self.writer.add_scalar('Return', self.current_return, self.episode_count)
            
            self.current_return = 0  # 重置当前回合的回报

            # 保存模型和绘制回报图像
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
        plt.title('SAC on MazeEnv')
        plt.savefig(self.return_file.replace('.json', '.png'))
        plt.close()

def train():
    log_dir = "sac_logs/"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return_file = os.path.join(log_dir, "returns.json")
    model_path = os.path.join(model_dir, "latest_model")  # 基础路径，无需扩展名
    tensorboard_log = os.path.join(log_dir, "tensorboard")

    env = make_vec_env(lambda: MazeEnv(headless=False), n_envs=1)

    sac_params = {
        'learning_rate': 3e-4,
        'buffer_size': 1000000,
        'learning_starts': 100,
        'batch_size': 256,
        'tau': 0.005,
        'gamma': 0.99,
        'train_freq': 10,
        'gradient_steps': 1,
        'action_noise': None,
        'replay_buffer_class': None,
        'replay_buffer_kwargs': None,
        'optimize_memory_usage': False,
        'ent_coef': 'auto',
        'target_update_interval': 100,
        'target_entropy': 'auto',
        'use_sde': False,
        'sde_sample_freq': -1,
        'use_sde_at_warmup': False,
        'stats_window_size': 100,
        'tensorboard_log': tensorboard_log,
        'policy_kwargs': {
            'net_arch': [256, 128, 128, 16],
            'activation_fn': nn.Tanh,
            'features_extractor_class': FlattenExtractor,
            'features_extractor_kwargs': {},
            'normalize_images': True,
            'optimizer_class': torch.optim.Adam,
            'optimizer_kwargs': {}
        },
        'verbose': 1,
        'seed': None,
        'device': 'cuda',
        '_init_setup_model': True
    }

    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip with new parameters")
        new_model = SAC('MlpPolicy', env, **sac_params)
        old_model = SAC.load(model_path + ".zip", env=env)
        new_model.policy.load_state_dict(old_model.policy.state_dict())
    else:
        print("No existing model found, training a new model.")
        new_model = SAC('MlpPolicy', env, **sac_params)

    callback = CustomCallback(return_file, model_path, check_freq=10, log_dir=tensorboard_log)

    new_model.learn(total_timesteps=1e7, callback=callback)

if __name__ == "__main__":
    train()
