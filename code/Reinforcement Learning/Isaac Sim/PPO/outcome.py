
import os
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from env import MazeEnv

def evaluate_trained_model(model_path, n_eval_episodes=10):
    # 创建环境
    env = MazeEnv(headless=False)
    
    # 加载模型
    model = PPO.load(model_path, env=env)
    
    # 评估模型，使用确定性策略
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True, return_episode_rewards=False)
    
    print(f"Mean reward: {mean_reward} +/- {std_reward}")
    
    return mean_reward, std_reward

if __name__ == "__main__":
    model_path = "ppo_logs/models/latest_model.zip"  # 替换为你的模型路径
    evaluate_trained_model(model_path, n_eval_episodes=1000)
