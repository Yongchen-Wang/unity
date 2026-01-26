import os
import numpy as np
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
# from gym_env import DiscreteMazeEnv  # 确保导入正确的环境类
from real_env import DiscreteMazeEnv
import time

def evaluate_model(model_path, num_episodes=10):
    # 创建和包装环境
    def make_env():
        env = DiscreteMazeEnv(
            grid_map_path='C:/Users/maoyudong/Desktop/PPO/PPO_TD3_DDPG_SAC/discrete_grid_map.npy',
            original_image_path='C:/Users/maoyudong/Desktop/PPO/PPO_TD3_DDPG_SAC/map.png',
            render_mode='human' , # 设置为 'none'，你可以根据需要调整
            seed=0
        )
        env = Monitor(env)  # 使用 Monitor wrapper 记录额外信息
        return env
    
    # 使用 DummyVecEnv 包装环境
    env = DummyVecEnv([make_env])
    
    # 加载模型
    model = A2C.load(model_path, env=env)
    
    # 开始评估
    episode_rewards = []
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=False)
            obs, rewards, dones, infos = env.step(action)
            total_reward += rewards[0]
            done = dones[0]  # 从 VecEnv 返回的 dones 是一个数组，取第一个元素
            
            # 如果你想要看到渲染结果，可以取消下面的注释
            env.envs[0].render()
            # time.sleep(0.1)

        episode_rewards.append(total_reward)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
    
    env.close()
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward}")

if __name__ == "__main__":
    model_path = "a2c_logs/models/latest_model.zip"  # 你的模型路径
    evaluate_model(model_path, num_episodes=100)  # 评估10个episode
