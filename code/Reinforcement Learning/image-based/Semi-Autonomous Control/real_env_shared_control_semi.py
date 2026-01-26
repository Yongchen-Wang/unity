
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
import time
import random
from sensapex import UMP
import math

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
        self.navigation_done = False  # 添加此属性以跟踪导航是否完成
        self.start_pos = (900,900)
        self.goal_pos = (900,900)
        self.current_pos = (900,900)

        self.ump = UMP.get_ump()
        self.manipulator_left = self.ump.get_device(1)

#####共享控制
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if not self.navigation_done:  # 如果导航未完成，重置位置

            # # RL 1
            # self.start_pos = (250, 250)  # 随机选择起点
            # self.goal_pos = (1400, 1200)  # 随机选择终点

            # # RL 2
            # self.start_pos = (1400, 1200)  # 随机选择起点
            # self.goal_pos = (200, 1500)  # 随机选择终点

            # RL 3
            self.start_pos = (200, 1500)  # 随机选择起点
            self.goal_pos = (1400, 300)  # 随机选择终点

            # maze1
            # self.start_pos = (250, 250)
            # self.goal_pos = (1100, 1200) 
            
            # #maze2
            # self.start_pos = (1400, 300)
            # self.goal_pos = (600, 1650)  

            # #maze3
            # self.start_pos = (1200, 1600)
            # self.goal_pos = (500, 300) 

            self.goto_position(self.start_pos)
            self.current_pos = self.start_pos
            self.step_num = 0
            self.collision_num = 0
            print("start_pos", self.start_pos, "goal_pos", self.goal_pos)
        return np.concatenate((np.array(self.current_pos, dtype=np.int32), np.array(self.start_pos, dtype=np.int32), np.array(self.goal_pos, dtype=np.int32))), {}



    def step(self, action):
        truncated = False
        self.step_num += 1
        print("self.step_num",self.step_num)
        
        # 将离散动作转换为x和y的位移
        dx, dy = self._map_action_to_movement(action)
        
        target_position = self.current_pos + np.array([dx, dy])

        if self._is_valid(target_position):
            self.goto_position(target_position)
            self.current_pos = self.get_position()
        else:
            self.current_pos=self.current_pos
            self.collision_num += 1
            # print("碰撞",self.collision_num)
        
        observation = np.concatenate((np.array(self.current_pos, dtype=np.int32), np.array(self.start_pos, dtype=np.int32), np.array(self.goal_pos, dtype=np.int32)))
        done = (abs(self.current_pos[0] - self.goal_pos[0]) <= 20) and (abs(self.current_pos[1] - self.goal_pos[1]) <= 20)
        if done:
            self.navigation_done=True
            print("done")
            
        
        if self.step_num >= self.max_episode_length:
            truncated = True
            print("truncated")
        
        reward = self.calculate_reward(target_position, done)
        info = {
            "step": self.step_num,
            "collision_num": self.collision_num
        }
        time.sleep(0.02)
        return observation, reward, done, truncated, info


    def get_position(self):
        x, y,z,w = self.manipulator_left.get_pos()
        # 转换坐标系，将机器人的坐标系映射到图像网格坐标系
        x_converted = 1800 * y / 18000  # 机器人Y轴对应图像网格的X轴
        y_converted = 1800 * x / 18000  # 机器人X轴对应图像网格的Y轴

        # 向下按十取整
        x_converted = (np.floor(x_converted / 10) * 10).astype(int)
        y_converted = (np.floor(y_converted / 10) * 10).astype(int)

        # print("x_converted,y_converted",x_converted,y_converted)
        return np.array([x_converted, y_converted])
    

    def get_position_no_convert(self):
        x, y,z,w = self.manipulator_left.get_pos()

        return np.array([x, y])

    def goto_position(self, position, speed=10000):
        x, y = position
        # 将图像网格坐标系映射到机器人的坐标系
        y_robot = 18000 * x / 1800  # 图像网格的X轴对应机器人的Y轴
        x_robot = 18000 * y / 1800  # 图像网格的Y轴对应机器人的X轴

        y_robot = np.floor(y_robot).astype(int)
        x_robot = np.floor(x_robot).astype(int)
        print("position",position)
        self.manipulator_left.goto_pos((x_robot, y_robot, 11000,10000), speed=speed)
        time.sleep(0.01)


    def _map_action_to_movement(self, action):
        # 根据仿真环境中的映射关系，将动作转换为实际的位移
        dx, dy = 10 * np.array(self.boundary_points[action])
        return dx, dy  # 正常返回dx和dy，而不是错误的覆盖操作

   

    def calculate_reward(self,target_position,done):
        reward = 0
        if not self._is_valid(target_position):
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
        valid_positions = [(i, j) for i in range(0, self.n_rows, 100) for j in range(0, self.n_cols, 100) if self.grid_map[i, j] == 0]
        start = random.choice(valid_positions)
        return start

    def render(self, mode='human'):
        pass

    def close(self):
        # if self.camera.isOpened():
        #     self.camera.release()
        cv2.destroyAllWindows()

    def seed(self, seed=None):
        np.random.seed(seed)
        random.seed(seed)

if __name__ == "__main__":
    env = DiscreteMazeEnv(
        grid_map_path='C:/Users/maoyudong/Desktop/PPO/8.10/discrete_grid_map.npy',
        original_image_path='C:/Users/maoyudong/Desktop/PPO/8.10/map.png',
        render_mode='human'  # 修正拼写错误
    )
 

