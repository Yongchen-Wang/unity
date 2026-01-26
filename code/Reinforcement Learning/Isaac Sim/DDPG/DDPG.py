import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import json

from DDPG_env import MazeEnv

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)

class ActorNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 128)
        self.action_head = torch.nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return torch.tanh(self.action_head(x))

class CriticNet(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, 1024)
        self.fc2 = torch.nn.Linear(1024, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.q_value_head = torch.nn.Linear(64, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        return self.q_value_head(x)

class DDPG:
    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, tau, device):
        self.actor = ActorNet(state_dim, action_dim).to(device)
        self.critic = CriticNet(state_dim, action_dim).to(device)
        self.target_actor = ActorNet(state_dim, action_dim).to(device)
        self.target_critic = CriticNet(state_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, noise_scale):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        action = self.actor(state).cpu().detach().numpy()[0]
        action += noise_scale * np.random.randn(action.shape[0])
        return np.clip(action, -1, 1)

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        states = torch.tensor(states, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float).view(-1, 1).to(self.device)

        # Update critic
        next_actions = self.target_actor(next_states)
        target_q_values = rewards + self.gamma * self.target_critic(next_states, next_actions) * (1 - dones)
        critic_loss = F.mse_loss(self.critic(states, actions), target_q_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, actor_path, critic_path):
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
        self.target_actor.load_state_dict(torch.load(actor_path))
        self.target_critic.load_state_dict(torch.load(critic_path))

if __name__ == '__main__':
    lr_actor = 1e-5
    lr_critic = 1e-5
    num_episodes = 1000
    gamma = 1
    tau = 0.001
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 512
    initial_noise_scale =0.5
    final_noise_scale=0.01
    noise_decay = 0.005 # 每个episode后噪声衰减的比例
    device = torch.device("cuda")

    env = MazeEnv(headless=False)
    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)
    replay_buffer = ReplayBuffer(buffer_size)
    state_dim = 4  # 两个物体的二维坐标
    action_dim = 2
    agent = DDPG(state_dim, action_dim, lr_actor, lr_critic, gamma, tau, device)

    # 加载之前保存的模型
    actor_model_path = "best_actor_model.pth"
    critic_model_path = "best_critic_model.pth"
    agent.load(actor_model_path, critic_model_path)

    noise_count=0
    return_list = []
    noise_scale = initial_noise_scale
    return_file = "return_list.json"

    with tqdm(total=num_episodes, desc='Training') as pbar:
        for i_episode in range(num_episodes):
            count = 0
            noise_count+=1
            state = env.reset()
            episode_return = 0
            done = False
            # while not done:
            while not done and count <= 1000:
                count += 1
                action = agent.take_action(state, noise_scale)
                next_state, reward, done, _ = env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward

                if replay_buffer.size() > minimal_size:
                    agent.update(replay_buffer, batch_size)
                if done:
                    print("done")

            return_list.append(episode_return)
            print(f"Episode {i_episode + 1}: Return {episode_return}")  # 打印每个回合的总奖励

            # 保存当前模型
            agent.save(actor_model_path, critic_model_path)
            print(f"Model saved after episode {i_episode + 1}")

            # 保存return_list到文件
            with open(return_file, 'w') as f:
                json.dump(return_list, f)

            # 衰减噪声
            if noise_count>=50:
                # if noise_scale >= final_noise_scale:
                #     noise_scale = noise_scale-noise_decay
                noise_scale=final_noise_scale

            # if noise_scale >= final_noise_scale:
            #     noise_scale = noise_scale-noise_decay
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(10)

def plot_returns(file_path):
    with open(file_path, 'r') as f:
        return_list = json.load(f)

    plt.plot(return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DDPG on MazeEnv')
    plt.show()

# 在需要绘制学习曲线时，调用plot_returns函数
# plot_returns(return_file)
