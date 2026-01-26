'''
./python3 tensorboard --logdir=/home/ym/.local/share/ov/pkg/isaac-sim-4.0.0/ppo_logs/tensorboard
'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
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
    def __init__(self, return_file, model_path, check_freq, log_dir, verbose=0, initial_ent_coef=0.01, final_ent_coef=0.00, schedule_timesteps=1e6):
        super(CustomCallback, self).__init__(verbose)
        self.return_file = return_file
        self.model_path = model_path
        self.check_freq = check_freq
        self.episode_count = 0
        self.return_list = []
        self.current_return = 0
        self.writer = SummaryWriter(log_dir)
        self.initial_ent_coef = initial_ent_coef
        self.final_ent_coef = final_ent_coef
        self.schedule_timesteps = schedule_timesteps
        self.ent_coef = initial_ent_coef
        if os.path.exists(return_file):
            with open(return_file, 'r') as f:
                self.return_list = json.load(f)

    def _on_step(self) -> bool:
        # 更新当前回合的回报
        self.current_return += self.locals['rewards'][0]

        # 动态调整熵系数
        progress = min(1.0, self.num_timesteps / self.schedule_timesteps)
        self.ent_coef = self.initial_ent_coef + progress * (self.final_ent_coef - self.initial_ent_coef)
        self.model.ent_coef = self.ent_coef

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
        plt.title('PPO on MazeEnv')
        plt.savefig(self.return_file.replace('.json', '.png'))
        plt.close()

def train():
    log_dir = "ppo_logs/"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return_file = os.path.join(log_dir, "returns.json")
    model_path = os.path.join(model_dir, "latest_model")  # 基础路径，无需扩展名
    tensorboard_log = os.path.join(log_dir, "tensorboard")

    env = make_vec_env(lambda: MazeEnv(headless=False), n_envs=1)

    ppo_params = {
        'learning_rate': 3e-4,  # 可以尝试降低学习率，例如1e-4
        'n_steps': 2048,  # 可以减少步数，例如2048
        'batch_size': 128,  # 可以尝试减小批量大小，例如512
        'n_epochs': 10,  # 可以减少训练的epoch数，例如10
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,  # 可以尝试减小clip范围，例如0.1
        'clip_range_vf': None,
        'normalize_advantage': True,
        'ent_coef': 0.001,  # 可以减小熵系数，例如0.01
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': 4,
        'rollout_buffer_class': None,
        'rollout_buffer_kwargs': None,
        'target_kl': None,
        'stats_window_size': 100,
        'tensorboard_log': tensorboard_log,
        'policy_kwargs': {
            'net_arch': [256,128,128,16],
            'activation_fn': nn.Tanh,
            'ortho_init': True,
            'log_std_init':-10,
            'full_std': True,
            'use_expln': False,
            'squash_output': False,
            # 'features_extractor_class': FlattenExtractor,
            'features_extractor_kwargs': {},
            'share_features_extractor': True,
            'normalize_images': True,
            # 'optimizer_class': torch.optim.Adam,
            'optimizer_kwargs': {}
        },
        'verbose': 1,
        'seed': None,
        'device': 'cuda',
        '_init_setup_model': True
    }

    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip with new parameters")
        new_model = PPO('MlpPolicy', env, **ppo_params)
        old_model = PPO.load(model_path + ".zip", env=env)
        new_model.policy.load_state_dict(old_model.policy.state_dict())
    else:
        print("No existing model found, training a new model.")
        new_model = PPO('MlpPolicy', env, **ppo_params)

    callback = CustomCallback(return_file, model_path, check_freq=100, log_dir=tensorboard_log, initial_ent_coef=0.001, final_ent_coef=0.000, schedule_timesteps=1e6)

    new_model.learn(total_timesteps=1e7, callback=callback)

if __name__ == "__main__":
    train()
