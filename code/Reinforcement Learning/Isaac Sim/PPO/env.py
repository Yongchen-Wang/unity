import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
from stable_baselines3.common.env_checker import check_env

class MazeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, skip_frame=3, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, max_episode_length=256, seed=0, headless=True, resolution=256):
        super().__init__()
        self.headless = headless
        self.skip_frame = skip_frame
        self.physics_dt = physics_dt
        self.resolution = resolution
        self.max_episode_length = max_episode_length
        self.step_num = 0
        self.collision_num = 0
        self.seed(seed)
        self.target_step_time = 1.0 / 60

        self.initialize_simulation(physics_dt, rendering_dt)

        # self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1, -1,-1,-1]), high=np.array([1, 1, 1, 1]), dtype=np.float32)
        self.reset() 

    def initialize_simulation(self, physics_dt, rendering_dt):
        from omni.isaac.kit import SimulationApp
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0, "multi_gpu": True})  # 启用multi_gpu
        from omni.isaac.core import World
        from omni.usd import get_context
        from omni.isaac.core.physics_context import PhysicsContext
        from pxr import UsdPhysics, PhysxSchema, Usd, Sdf

        # Load the stage
        maze_asset_path = "/home/ym/model/v6.usd"
        get_context().open_stage(maze_asset_path)
        self._simulation_app.update()
        from omni.isaac.core.utils.stage import is_stage_loading
        while is_stage_loading():
            self._simulation_app.update()

        # Initialize the world
        self.my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt)
        self.physics_context = PhysicsContext(physics_dt=physics_dt)
        self.physics_context.set_gravity(0.0)

        # Enable GPU dynamics and broadphase on the physics scene
        stage = get_context().get_stage()
        scene_prim_path = "/World/PhysicsScene"
        scene_prim = stage.GetPrimAtPath(scene_prim_path)
        if not scene_prim:
            scene_prim = UsdPhysics.Scene.Define(stage, scene_prim_path).GetPrim()

        # Create and set attributes
        if not scene_prim.GetAttribute('physxScene:gpuDynamicsEnabled'):
            gpu_dynamics_enabled_attr = scene_prim.CreateAttribute('physxScene:gpuDynamicsEnabled', Sdf.ValueTypeNames.Bool, custom=True)
            gpu_dynamics_enabled_attr.Set(True)
        else:
            scene_prim.GetAttribute('physxScene:gpuDynamicsEnabled').Set(True)

        if not scene_prim.GetAttribute('physxScene:gpuBroadphaseEnabled'):
            gpu_broadphase_enabled_attr = scene_prim.CreateAttribute('physxScene:gpuBroadphaseEnabled', Sdf.ValueTypeNames.Bool, custom=True)
            gpu_broadphase_enabled_attr.Set(True)
        else:
            scene_prim.GetAttribute('physxScene:gpuBroadphaseEnabled').Set(True)

        self.my_world.play()

        # Initialize articulation and other components
        self.initialize_components()


    def initialize_components(self):
        from omni.isaac.core.prims import RigidPrim, GeometryPrim
        from omni.isaac.sensor import ContactSensor

        self.agent = GeometryPrim(prim_path="/World/Cylinder")
        self.target = RigidPrim(prim_path="/World/Cube")
        self.agent.initialize()
        self.agent.set_world_pose(np.array([0.015, 0.025, 0.5]))


        self.target.initialize()
        self.target.set_world_pose(np.array([0.015, 0.025, 0.03]),np.array([1,0,0,0]))




        self.agent_home_pose = self.agent.get_world_pose()
        self.target_home_pose = self.target.get_world_pose()


    
    def take_continuous_action(self, action):
        import omni.isaac.core.utils.rotations as rotations_utils

        step_length = 0.001
        dt = self.target_step_time
        current_target_position = self.target.get_world_pose()[0]
        current_target_orientatioin=self.target.get_world_pose()[1]

        current_agent_position = self.agent.get_world_pose()[0]
        # print('前_小球',current_target_position)
        # print('前_杆子',current_agent_position)
        

        # Scale action to appropriate step size
        next_position = current_agent_position + np.array([action[0], action[1], 0.0]) * step_length

        distance_to_center = np.linalg.norm(np.array([next_position[0],next_position[1]]) - np.array([current_target_position[0], current_target_position[1]]))



        yaw =action[1]/action[0]

        new_euler_angles = [0, 0, yaw]
        new_orientation_quat = rotations_utils.euler_angles_to_quat(new_euler_angles,extrinsic=False)

 
        if current_target_position[2] > 0.0305 or current_target_position[2]<0.0295:
            current_target_position = [current_target_position[0],current_target_position[1],0.03]            
            self.target.set_world_pose(current_target_position)






        if distance_to_center<=0.002:
            self.agent.set_world_pose(next_position)

            dx = next_position[0]-current_target_position[0]
            dy = next_position[1]-current_target_position[1]

            linear_velocity = np.array([dx, dy, 0]) / dt
            self.target.set_world_pose(current_target_position,new_orientation_quat)


            self.target.set_linear_velocity(linear_velocity)


            self.target_position=next_position
            # print("next_position",next_position)
        else:
            self.agent.set_world_pose(current_agent_position)

            dx = current_agent_position[0]-current_target_position[0]
            dy = current_agent_position[1]-current_target_position[1]

            linear_velocity = np.array([dx, dy, 0]) / dt
            
            self.target.set_world_pose(current_target_position,current_target_orientatioin)

            self.target.set_linear_velocity(linear_velocity)

            self.target_position=current_agent_position
            # print("current_agent_position",current_agent_position)

            




    def check_action_success(self):


        # 获取动作执行完毕后的实际位置和姿态
        actual_position = self.target.get_world_pose()[0]
        target_position=self.agent.get_world_pose()[0]

        # print("后_小球",actual_position)
        # print("后_杆子",target_position)

        # 计算位置和姿态的误差
        position_error_x=np.linalg.norm(actual_position[0] - target_position[0])
        position_error_y=np.linalg.norm(actual_position[1] - target_position[1])
        # 检查误差是否在阈值范围内
        position_threshold = 0.0002

        if position_error_x < position_threshold and position_error_y < position_threshold:
            return True
        else:
            return False


    def step(self, action):
        self.step_num += 1
        self.take_continuous_action(action)
        self.simpleStep()
        action_success = self.check_action_success()
        # print(action_success)

        current_target_position, current_agent_position = self.get_observations()
        done, reward, truncated = self.calculate_reward(current_target_position, current_agent_position, action_success)

        
        observation = np.concatenate( [current_target_position[:2],current_agent_position[:2]])
        info = {"step": self.step_num}

        return observation, reward, done, truncated, info

    def get_observations(self):
        current_target_position = self.target.get_world_pose()[0]
        current_agent_position = self.agent.get_world_pose()[0]
        return current_target_position, current_agent_position

    def calculate_reward(self, current_target_position, current_agent_position, action_success):
        done = False
        reward = 0  # 每一步都要扣一分
        des = np.array([0.121, 0.165, 0.03])
        norm_endpoint = np.linalg.norm(current_target_position - des)

        if not action_success:
            reward -= 1
            self.collision_num+=1

        truncated = False
        if current_target_position[0] >= 0.2 or current_target_position[0] <= 0 or current_target_position[1] >= 2 or current_target_position[1] <= 0 or self.step_num > 2e4:
            print("collision_num:",self.collision_num,"step_num:",self.step_num,"rate",(self.collision_num/self.step_num))
            truncated = True
            reward -= 5000

        if norm_endpoint != 0:
            reward += -50 * norm_endpoint

        if 0.118 <= current_target_position[0] <= 0.126 and 0.16 <= current_target_position[1] <= 0.175:
            done = True
            reward += 10000
            print("collision_num:",self.collision_num,"step_num:",self.step_num,"rate",(self.collision_num/self.step_num))
            print("Done")

        return done, reward, truncated



    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent.set_world_pose(np.array([0.015, 0.025, 0.5]))
        self.target.set_world_pose(np.array([0.015, 0.025, 0.03]),np.array([1,0,0,0]))

        self.target.set_linear_velocity([0,0,0])
        self.step_num = 0
        self.collision_num=0
        self.target_position = None
        current_target_position, current_agent_position = self.get_observations()

        observation = np.concatenate( [current_target_position[:2],current_agent_position[:2]])
        return observation, {}

    def render(self, mode="human"):
        self.my_world.render()

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        np.random.seed(seed)

    def simpleStep(self):
        self.my_world.step()

if __name__ == "__main__":
    env = MazeEnv(headless=False)
    env.reset()
    
    # 固定的前进动作，第二个元素为0表示不旋转
    action = [0.1,0]
    
    for _ in range(5000):
        observation, reward, done, truncated, info = env.step(action)
        env.render()
        if done:
            break
    
    env.close()
#     env = MazeEnv(headless=False)

# # 检查环境
#     print(check_env(env))