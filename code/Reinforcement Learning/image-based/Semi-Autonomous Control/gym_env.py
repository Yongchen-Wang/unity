
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import random

class DiscreteMazeEnv(gym.Env):
    metadata = {"render.modes": ["human", "none"]}

    def __init__(self, grid_map_path, original_image_path, max_episode_length=2e4, seed=0, render_mode='human'):
        super().__init__()
        self.grid_map = np.load(grid_map_path)
        if self.grid_map is None:
            raise ValueError(f"Failed to load the grid map file {grid_map_path}.")
        self.original_image = cv2.imread(original_image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to read the image file {original_image_path}.")
        
        self.n_rows, self.n_cols = self.grid_map.shape
        self.max_episode_length = max_episode_length
        self.seed(seed)
        
        self.render_mode = render_mode
        
        self.boundary_points = [(x, y) for x in range(-5, 6) for y in range(-5, 6) if x == -5 or x == 5 or y == -5 or y == 5]
        self.action_space = spaces.Discrete(len(self.boundary_points))
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0]), 
            high=np.array([self.n_rows - 1, self.n_cols - 1, self.n_rows - 1, self.n_cols - 1, self.n_rows - 1, self.n_cols - 1]), 
            dtype=np.int32
        )

        self.start_pos = (900,900)
        self.goal_pos = (900,900)
        self.current_pos = None
        # self.reset() 

    # def reset(self, seed=None, options=None):
    #     super().reset(seed=seed)
    #     self.start_pos = self._get_random_start()  # 随机选择起点
    #     self.goal_pos = self._get_random_goal()  # 随机选择终点
    #     # self.start_pos = (900,900)
    #     # self.goal_pos = (1400,300) # 随机选择终点
    #     self.current_pos = self.start_pos
    #     self.step_num = 0
    #     self.collision_num = 0
    #     return np.concatenate((np.array(self.current_pos, dtype=np.int32), np.array(self.start_pos, dtype=np.int32), np.array(self.goal_pos, dtype=np.int32))), {}
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        

        # self.start_pos = self.goal_pos  # 将上一次的目标点设为新的起始点
        # self.goal_pos = self._get_random_goal()  # 随机选择新的终点

        self.start_pos = (250, 250)  # 随机选择起点
        self.goal_pos = (1400, 1200)  # 随机选择终点

        print("start_pos", self.start_pos, "goal_pos", self.goal_pos)

        self.current_pos = self.start_pos
        self.step_num = 0
        self.collision_num = 0
        time.sleep(2)
        return np.concatenate((np.array(self.current_pos, dtype=np.int32), np.array(self.start_pos, dtype=np.int32), np.array(self.goal_pos, dtype=np.int32))), {}
    def step(self, action):
        truncated = False
        self.step_num += 1
        
        # 将离散动作转换为x和y的位移
        dx, dy = 10*np.array(self.boundary_points[action])
        
        new_pos = [self.current_pos[0] + dx, self.current_pos[1] + dy]
        new_pos[0] = np.clip(new_pos[0], 0, self.n_rows - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.n_cols - 1)

        move_distance = np.linalg.norm(np.array([dx, dy]))
        if self._is_valid(new_pos):
            self.current_pos = tuple(new_pos)
        else:
            self.collision_num += 1

        print("self.current_pos",self.current_pos)
        time.sleep(0.1)
        observation = np.concatenate((np.array(self.current_pos, dtype=np.int32), np.array(self.start_pos, dtype=np.int32), np.array(self.goal_pos, dtype=np.int32)))
        done = (abs(self.current_pos[0] - self.goal_pos[0]) <= 50) and (abs(self.current_pos[1] - self.goal_pos[1]) <= 50)
        if done:
            print("done")
        
        if self.step_num >= self.max_episode_length:
            truncated = True
            print("truncated")
        
        reward = self.calculate_reward(new_pos, done,move_distance)
        info = {
            "step": self.step_num,
            "collision_num": self.collision_num
        }

        return observation, reward, done, truncated, info

    def calculate_reward(self, new_pos, done,move_distance):
        distance_to_goal = np.linalg.norm(np.array(self.current_pos) - np.array(self.goal_pos))
        reward = 0
        reward -= 0.1 * move_distance/50
        reward -= 0.005 * distance_to_goal

        if not self._is_valid(new_pos):
            reward -= 1

        if done:
            reward += 1000
        
        return reward

    def _is_valid(self, pos):
        x, y = pos
        if 0 <= x < self.n_rows and 0 <= y < self.n_cols:
            return self.grid_map[x, y] == 0  # 修改为0表示可通行
        return False

    def _get_random_goal(self):
        valid_positions = [(i, j) for i in range(0, self.n_rows, 100) for j in range(0, self.n_cols, 100) if self.grid_map[i, j] == 0]
        goal = random.choice(valid_positions)
        return goal

    def _get_random_start(self):
        valid_positions = [(i, j) for i in range(0, self.n_rows, 50) for j in range(0, self.n_cols, 50) if self.grid_map[i, j] == 0]
        start = random.choice(valid_positions)
        return start

    def render(self, mode='human'):
        if self.render_mode == 'human':
            img = self.original_image.copy()
            
            # 标记起点为绿色
            start_x, start_y = self.start_pos
            cv2.circle(img, (start_y, start_x), 50, (0, 255, 0), -1)
            
            # 标记终点为蓝色
            goal_x, goal_y = self.goal_pos
            cv2.circle(img, (goal_y, goal_x), 50, (255, 0, 0), -1)
            
            # 标记当前 agent 位置为红色
            x, y = self.current_pos
            cv2.circle(img, (y, x), 50, (0, 0, 255), -1)
            
            img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
            cv2.imshow('Maze', img)
            cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

    # def visualize_all_goals(self):
    #     img = self.original_image.copy()
        
    #     # 获取所有有效的终点位置
    #     valid_positions = [(i, j) for i in range(0, self.n_rows, 50) for j in range(0, self.n_cols, 50) if self.grid_map[i, j] == 0]
        
    #     # 在图像上标记所有有效的终点位置
    #     for goal_x, goal_y in valid_positions:
    #         cv2.circle(img, (goal_y, goal_x), 20, (255, 0, 0), -1)  # 用蓝色的小圆圈标记所有可能的终点
        
    #     img = cv2.resize(img, (500, 500), interpolation=cv2.INTER_NEAREST)
    #     cv2.imshow('All Possible Goals', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

if __name__ == "__main__":
    env = DiscreteMazeEnv(
        grid_map_path='C:/Users/maoyudong/Desktop/PPO/8.10/discrete_grid_map.npy',
        original_image_path='C:/Users/maoyudong/Desktop/PPO/8.10/map.png',
        render_mode='human'  # 修正拼写错误
    )

    # env.visualize_all_goals()
    observation, _ = env.reset()
    
    done = False
    truncated = False  # 添加初始化 truncated 变量
    while not done and not truncated:  # 修改循环条件为 "and" 以避免无限循环
        action = env.action_space.sample()
        observation, reward, done, truncated, _ = env.step(action)
        env.render()  # 始终调用 render 方法
        time.sleep(0.1)

    env.close()



