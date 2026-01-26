
'''
./python3 tensorboard --logdir=/home/ym/.local/share/ov/pkg/isaac-sim-4.0.0/dqn_logs/tensorboard

'''
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
from gym_env import DiscreteMazeEnv
from stable_baselines3.dqn.policies import DQNPolicy

class CustomDQNPolicy(DQNPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
            net_arch=[512, 256, 128, 64]
        )

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
                # 直接创建新文件保存模型
                save_path = f"{self.model_path}_ep{self.episode_count}.zip"
                self.model.save(save_path)
                print(f"Model saved at {save_path}")
                # 添加保存经验回放池的代码
                replay_buffer_path = f"{self.model_path}_ep{self.episode_count}_replay_buffer.pkl"
                self.model.save_replay_buffer(replay_buffer_path)
                print(f"Replay buffer saved at {replay_buffer_path}")
        self.training_env.render(mode='none')

        return True

    def plot_returns(self):
        with open(self.return_file, 'w') as f:
            json.dump(self.return_list, f)

        plt.plot(self.return_list)
        plt.xlabel('Episodes')
        plt.ylabel('Returns')
        plt.title('DQN on MazeEnv')
        plt.savefig(self.return_file.replace('.json', '.png'))
        plt.close()

def train():
    log_dir = "dqn_logs/"
    os.makedirs(log_dir, exist_ok=True)
    model_dir = os.path.join(log_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    return_file = os.path.join(log_dir, "returns.json")
    model_path = os.path.join(model_dir, "latest_model")  # 基础路径，无需扩展名
    tensorboard_log = os.path.join(log_dir, "tensorboard")

    env = make_vec_env(lambda: DiscreteMazeEnv(
        grid_map_path='C:/Users/maoyudong/Desktop/PPO/8.10/discrete_grid_map.npy',
        original_image_path='C:/Users/maoyudong/Desktop/PPO/8.10/map.png',
        render_mode='none'
    ), n_envs=1)

    # 新的超参数
    new_dqn_params = {
        'learning_rate': 5e-5,  # 更新的学习率
        'buffer_size': 500000,
        'learning_starts': 10000,
        'batch_size': 512,
        'tau': 1.0,
        'gamma': 0.99,
        'train_freq': 50,
        'gradient_steps': 1,
        'target_update_interval': 20000,
        'exploration_fraction': 0.8,
        'exploration_initial_eps': 0.5,
        'exploration_final_eps': 0.05,
        'max_grad_norm': 10,
        'tensorboard_log': tensorboard_log,
        'verbose': 1,
        'device': 'cuda',
    }

    # 加载模型参数
    if os.path.exists(model_path + ".zip"):
        print(f"Loading model from {model_path}.zip with new parameters")
        new_model = DQN(CustomDQNPolicy, env, **new_dqn_params)
        old_model = DQN.load(model_path + ".zip", env=env)
        new_model.policy.load_state_dict(old_model.policy.state_dict())

        # 添加加载经验回放池的代码
        replay_buffer_path = f"{model_path}_replay_buffer.pkl"
        if os.path.exists(replay_buffer_path):
            new_model.load_replay_buffer(replay_buffer_path)
            print(f"Replay buffer loaded from {replay_buffer_path}")
        else:
            print("No replay buffer found, starting with an empty buffer.")
    else:
        print("No existing model found, training a new model.")
        new_model = DQN(CustomDQNPolicy, env, **new_dqn_params)

    callback = CustomCallback(return_file, model_path, check_freq=10000, log_dir=tensorboard_log)

    new_model.learn(total_timesteps=1e8, callback=callback)

if __name__ == "__main__":
    train()
