import random
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
import json

from DDPG_env import MazeEnv
from collections import deque

class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.action_head = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.fc1(x))
        x = torch.nn.functional.leaky_relu(self.fc2(x))
        x = torch.nn.functional.leaky_relu(self.fc3(x))
        return torch.tanh(self.action_head(x))

class DDPG:
    def __init__(self, state_dim, action_dim, device):
        self.actor = ActorNet(state_dim, action_dim).to(device)
        self.device = device

    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        return np.clip(action, -1, 1)

    def load(self, actor_path):
        self.actor.load_state_dict(torch.load(actor_path))

if __name__ == '__main__':
    lr_actor = 1e-5
    num_episodes = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = MazeEnv(headless=False)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    state_dim = 4  # 两个物体的二维坐标
    action_dim = 2
    agent = DDPG(state_dim, action_dim, device)

    # 加载之前保存的模型
    actor_model_path = "best_actor_model.pth"
    agent.load(actor_model_path)

    return_list = []

    for i_episode in range(num_episodes):
        state = env.reset()
        episode_return = 0
        done = False
        count = 0
        
        while not done and count <= 1000:
            count += 1
            action = agent.take_action(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_return += reward

        return_list.append(episode_return)
        print(f"Episode {i_episode + 1}: Return {episode_return}")  # 打印每个回合的总奖励

    # 保存return_list到文件
    return_file = "return_list_eval.json"
    with open(return_file, 'w') as f:
        json.dump(return_list, f)

    # 绘制学习曲线
    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG Evaluation on MazeEnv')
    plt.show()
