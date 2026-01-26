import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib.magnet import Cylinder
from numpy import pi
from magforce import getF
import cv2
import os

class MazeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, skip_frame=3, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, max_episode_length=256, seed=0, headless=True, resolution=256):
        super().__init__()
        self.headless = headless
        self.skip_frame = skip_frame
        self.physics_dt = physics_dt
        self.resolution = resolution
        self.initialize_simulation(physics_dt, rendering_dt)
        self.step_num = 0
        self.save_dir = '/home/ym/saved_images/'
        self.target_positions = set()  # 初始化目标位置集合
        self.mid_done = True
        self.mid_done2 = True

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=0.5, shape=(4,), dtype=np.float32)  # 两个物体的二维坐标

    def initialize_simulation(self, physics_dt, rendering_dt):
        from omni.isaac.kit import SimulationApp
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        from omni.isaac.core import World
        from omni.usd import get_context

        # Load the stage
        maze_asset_path = "/home/ym/model/v1.usda"
        get_context().open_stage(maze_asset_path)
        self._simulation_app.update()
        from omni.isaac.core.utils.stage import is_stage_loading
        while is_stage_loading():
            self._simulation_app.update()

        # Initialize the world
        self.my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt)
        self.my_world.play()
        # Initialize articulation and other components
        self.initialize_components()

    def initialize_components(self):
        from omni.isaac.core.prims import RigidPrim, GeometryPrim
        from omni.isaac.sensor import Camera

        self.agent = GeometryPrim(prim_path="/World/Cylinder")
        self.target = RigidPrim(prim_path="/World/Sphere")
        self.cam_01 = Camera(prim_path='/World/Camera', resolution=(self.resolution, self.resolution))
        density = 7900
        self.target.set_density(density)

        self.agent.initialize()
        self.agent.set_world_pose(np.array([0.05, 0.05, 0.1]))
        # self.agent.set_world_pose(np.array([0.18,0.05,0.1]))
        self.target.initialize()
        self.target.set_world_pose(np.array([0.05, 0.05,0.035]))
        # self.target.set_world_pose(np.array([ 0.18,0.05,0.035]))
        self.cam_01.initialize()

        self.agent_home_pose = self.agent.get_world_pose()
        self.target_home_pose = self.target.get_world_pose()

    def calculate_xy_vel(self, current_agent_position, current_target_position):
        demagnetizing_factor = 1/3  # sphere
        volume = 4 / 3 * pi * (0.5/1000) ** 3
        M_saturation = 480000  # Ms Co room temperature [A/m]

        sample = {'demagnetizing_factor': demagnetizing_factor,
                  'volume': volume,
                  'M_saturation': M_saturation}

        m1 = Cylinder(magnetization=[0, 0, 5e10], dimension=(1, 10),
                      position=current_agent_position, orientation=None)

        radius = 0.5 / 1000  # 半径转换为米
        density = 7900  # 密度 (kg/m³)
        mass = (4/3) * pi * (radius ** 3) * density  # 计算质量
        if np.linalg.norm(current_target_position-current_agent_position) > 0.001:
            F_m1 = getF(current_target_position, m1, sample)
            F_xy = np.array([F_m1[0], F_m1[1], 0])
            vel_xy = -F_xy / mass * self.physics_dt
            return vel_xy
        else:
            return np.array([0, 0, 0])

    def take_continuous_action(self, action):
        step_length = 0.01
        current_target_position = self.target.get_world_pose()[0]
        current_agent_position = self.agent.get_world_pose()[0]

        # Scale action to appropriate step size
        next_position = current_agent_position + np.array([action[0], action[1], 0.0]) * step_length

        distance_to_center = np.linalg.norm(np.array([next_position[0],next_position[1]]) - np.array([current_target_position[0], current_target_position[1]]))
        # if distance_to_center >= 0.02 or next_position[0] < 0.025 or next_position[0] > 0.335 or next_position[1] < 0.025 or next_position[1] > 0.335:
        if distance_to_center >= 0.02:
            next_position=current_agent_position

            self.agent.set_world_pose(next_position)
            current_agent_position = self.agent.get_world_pose()[0]
            current_target_position = self.target.get_world_pose()[0]

            current_target_vel_xy = self.calculate_xy_vel(current_agent_position, current_target_position)
            self.target.set_linear_velocity(current_target_vel_xy)
            return False

        else :  # 这个标准还需要判断一下
            next_position=next_position

            self.agent.set_world_pose(next_position)
            current_agent_position = self.agent.get_world_pose()[0]
            current_target_position = self.target.get_world_pose()[0]

            current_target_vel_xy = self.calculate_xy_vel(current_agent_position, current_target_position)
            self.target.set_linear_velocity(current_target_vel_xy)

            return True


    
    def step(self, action):
        self.step_num += 1
        action_success = self.take_continuous_action(action)
        current_target_position, current_agent_position = self.get_observations()

        done, reward, info = self.calculate_reward(current_target_position, current_agent_position, action_success)
        state = np.concatenate([np.round(current_agent_position[:2], 3), np.round(current_target_position[:2], 3)])  # 两个二维坐标拼接成四维状态

        self.simpleStep()

        return state, reward, done, info

    def get_observations(self):
        current_target_position = np.round(self.target.get_world_pose()[0], 3)
        current_agent_position = np.round(self.agent.get_world_pose()[0], 3)
        return current_target_position, current_agent_position

    def calculate_reward(self, current_target_position, current_agent_position,action_success):
        done = False
        info = True
        reward = -1  # 每一步都要扣一分
        des = np.array([0.34, 0.05, 0.035])
        norm_endpoint = np.linalg.norm(current_target_position - des)


        # target_pos_tuple = tuple(np.round(current_target_position, decimals=3))
        # if target_pos_tuple in self.target_positions:
        #     reward -= 10
        if not action_success:
            reward -= 10
        # if norm_endpoint >= 0.25:
        #     reward += -3
        # elif norm_endpoint>=0.15:
        #     reward += (2-20*norm_endpoint)
        # else :
        #     reward += (5-10*norm_endpoint)

        if norm_endpoint !=0 :
            reward += (1-20*norm_endpoint)

        if current_target_position[0]>=0.3 and current_target_position[0]<=0.34 and current_target_position[1]<=0.07:
            done = True
            reward += 1000
            print("Done")


        return done, reward, info

    # def update_target_positions(self, current_target_position):
    #     target_pos_tuple = tuple(np.round(current_target_position, decimals=3))
    #     self.target_positions.add(target_pos_tuple)

    def reset(self):
        self.target.set_world_pose(np.array([0.05, 0.05, 0.035]))
        # self.target.set_world_pose(np.array([0.18,0.05,  0.035]))
        self.target.set_linear_velocity(np.array([0, 0, 0]))
        self.agent.set_world_pose(np.array([0.05, 0.05, 0.1]))
        # self.agent.set_world_pose(np.array([0.18,0.05,  0.1]))
        
        self.target_positions = set()  # 重置目标位置集合
        self.mid_done = True
        self.mid_done2 = True

        current_target_position, current_agent_position = self.get_observations()
        state = np.concatenate([np.round(current_agent_position[:2], 3), np.round(current_target_position[:2], 3)])  # 两个二维坐标拼接成四维状态
        return state

    def render(self, mode="human"):
        self.my_world.render()
        pass

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        pass

    def simpleStep(self):
        self.my_world.step()

if __name__ == "__main__":
    env = MazeEnv(headless=False)
    env.reset()
    t = 0
    max_steps = 512
    for i in range(max_steps):
        action = np.random.uniform(-1, 1, size=(2,))
        state, reward, done, info = env.step(action)
        print(f"Step {i}, State: {state}, Reward: {reward}, Done: {done}")
        t += 1
        if done:
            break
    env.close()
