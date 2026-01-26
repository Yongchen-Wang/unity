import gym
from gym import spaces
import numpy as np
from scipy.spatial.transform import Rotation as R
from magpylib.magnet import Cylinder
from numpy import pi
from magforce import getF



####
# 闭包函数
####
def create_smart_one_time_printer():
    printed_messages = set()

    def printer(message):
        message = str(message)
        if message not in printed_messages:
            print(message)
            printed_messages.add(message)
    return printer


def quaternion_to_euler(quaternion):
    rotation = R.from_quat(quaternion)
    return rotation.as_euler('zyx', degrees=True)

def euler_to_quaternion(euler_angles):
    rotation = R.from_euler('zyx', euler_angles, degrees=True)
    return rotation.as_quat()

def adjust_rotation(quaternion, z_adjust_degrees):
    # 将四元数转换为欧拉角
    euler_angles = quaternion_to_euler(quaternion)

    # 在 Z 轴上减少 90 度
    euler_angles[1] -= z_adjust_degrees

    # 将调整后的欧拉角转换回四元数
    return euler_to_quaternion(euler_angles)

def quaternion_to_rotation_matrix(quaternion):
    """
    将四元数转换为旋转矩阵
    """
    # 提取四元数的各个部分
    w, x, y, z = quaternion
    xx, xy, xz = x * x, x * y, x * z
    yy, yz, zz = y * y, y * z, z * z
    wx, wy, wz = w * x, w * y, w * z

    # 计算旋转矩阵
    rotation_matrix = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return rotation_matrix

# Import necessary modules from omni.isaac

def create_transformation_matrix(translation, rotation):
    # 创建旋转矩阵
    R = np.array(rotation).reshape(3, 3)

    # 创建转换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T

def transform_from_gripper_to_link6(gripper_pos, gripper_orient_quat, translation):
    # 将四元数转换为旋转矩阵
    gripper_orient = quaternion_to_rotation_matrix(gripper_orient_quat)

    # 应用旋转矩阵到平移向量
    transformed_translation = np.dot(gripper_orient, translation)

    # 应用转换后的平移向量到 Link 6 的位置
    link6_pos = gripper_pos + transformed_translation

    return link6_pos




class MazeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, skip_frame=3, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, max_episode_length=256, seed=0, headless=True, resolution=256):
        super().__init__()
        self.headless = headless
        self.skip_frame = skip_frame
        self.physics_dt = physics_dt
        self.resolution = resolution
        self.initialize_simulation(physics_dt, rendering_dt)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict({
        'image': spaces.Box(low=0.0, high=255.0, shape=(self.resolution, self.resolution, 3), dtype=np.float32),
        })
        self.reset_counter = 0
        self.goal_position = np.array([-3.5, 3.5])
        self.explored_areas = set() 
        self.radius_of_ball = 0.005
        self.step_num = 0
        
        self.grid_size = 4 * self.radius_of_ball

        self.position_history = []  # 存储位置的历史记录
        self.no_change_count = 0  # 计数器


    def initialize_simulation(self, physics_dt, rendering_dt):
        from omni.isaac.kit import SimulationApp
  
        # Initialize the simulation application
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})

        from omni.isaac.core import World
        from omni.usd import get_context

        # Load the stage
        maze_asset_path = "/home/jialin/Documents/isaacFiles/simRLenvEasy.usda"
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
        from omni.isaac.core.prims import RigidPrim,GeometryPrim,RigidPrimView
        from omni.isaac.motion_generation.lula import RmpFlow
        from omni.isaac.core.articulations import Articulation
        
        from omni.isaac.motion_generation import ArticulationMotionPolicy
        from omni.isaac.sensor import Camera

        self.agent = RigidPrim("/maze1/agent")
        self.mag = RigidPrim(prim_path="/UnitreeZ1/magnet")
  
        self.target = GeometryPrim(prim_path="/World/target")
        self.articulation = Articulation(prim_path="/UnitreeZ1")

        self.cam_01 = Camera(prim_path='/World/camera/Camera',resolution=(self.resolution,self.resolution))
        self.cam_02 = Camera(prim_path='/World/camera/Camera_01',resolution=(self.resolution,self.resolution))
        self.cam_hand = Camera(prim_path='/UnitreeZ1/magnet/Camera',resolution=(self.resolution,self.resolution))
        density = 7900 
        self.agent.set_density(density)
        self.agent.initialize()
        self.mag.initialize()
        self.target.initialize()
        self.articulation.initialize()
        self.cam_01.initialize()
        self.cam_02.initialize()
        self.cam_hand.initialize()

        self.target_home_pose = self.target.get_world_pose()
        self.agent_home_pose = self.agent.get_world_pose()

    
        self.rmpflow = RmpFlow(
            robot_description_path = "/home/jialin/Documents/isaacFiles/z1/newLula.yaml",
            urdf_path ="/home/jialin/Documents/isaacFiles/z1/z1.urdf",
            rmpflow_config_path ="/home/jialin/Documents/isaacFiles/z1/rmp_config_z1.yaml",
            end_effector_frame_name = "link06",
            maximum_substep_size = 0.00334
        )
        self.articulation_rmpflow = ArticulationMotionPolicy(robot_articulation=self.articulation, motion_policy=self.rmpflow, default_physics_dt=self.physics_dt)

        self.homing()

        # self.printer = create_smart_one_time_printer()
        
    def calculate_xy_vel(self, current_mag_position, current_agent_position):

        demagnetizing_factor = 1/3             # sphere
        volume = 4 / 3 * pi * (7.5/1000) ** 3  
        M_saturation = 480000               # Ms Co room temperature [A/m]

        sample = {'demagnetizing_factor': demagnetizing_factor,
          'volume': volume,
          'M_saturation': M_saturation}
        
        m1 = Cylinder(magnetization=[0, 0, -1000], dimension=(20,15),
                    position= current_mag_position, orientation=None)

        radius = 7.5 / 1000  # 半径转换为米
        density = 7900  # 密度 (kg/m³)
        mass = (4/3) * pi * (radius ** 3) * density  # 计算质量

        F_m1 = getF(current_agent_position, m1, sample) *50

        F_xy = np.array([F_m1[0],F_m1[1],0])

        vel_xy = F_xy / mass * self.physics_dt

        # print(vel_xy)
            # 计算瞬时速度而不是加速度
            # 注意：这里假设 delta_t 是足够小的，以至于速度变化不会太剧烈
        
        return vel_xy
    
    def check_position_update(self,current_position ):
   
        if len(self.position_history) == 0 or np.array_equal(self.position_history[-1], current_position):
            self.position_history.append(current_position)
            self.no_change_count = 0  # 重置计数器
        else:
           self.no_change_count += 1  # 增加计数器

        # 保持位置历史记录的长度不超过10
        if len(self.position_history) > 300:
            self.position_history.pop(0)

        # 检查计数器是否达到10
        if self.no_change_count >= 300:
            return True  # 或者返回一个特定的标志
        else:
            return False


    def check_action_in_boundary(self,action,current_target_position):
        
        step_length = 0.001
        if action == 0:
            next_position = current_target_position+ np.array([step_length,0.0,0.0])
        elif action == 1:
            next_position = current_target_position+ np.array([-step_length,0.0,0.0])
        elif action == 2:
            next_position = current_target_position+ np.array([0.0,step_length,0.0])
        elif action == 3:
            next_position = current_target_position+ np.array([0.0,-step_length,0.0])

        distance_to_center = np.linalg.norm(next_position - np.array([0.3,0,0.13]))
        if distance_to_center <= 0.1:
            return (next_position, True)
        else:
            return (current_target_position, False)
        

    def take_discrete_action(self, action):
        current_target_position = self.target.get_world_pose()[0]
        

        
    
        next_pose, success = self.check_action_in_boundary(action, current_target_position)

        self.target.set_world_pose(next_pose)

        tolerance = 0.01
        while True:
            
            current_mag_position, current_mag_orient = self.mag.get_world_pose()[0], self.mag.get_world_pose()[1]
            gripper_pos, gripper_orient = self.target.get_world_pose()[0],self.target.get_world_pose()[1]
            if np.allclose(current_mag_position, gripper_pos, atol=tolerance):
                break

            translation = [0, 0, -0.08]  # 假设电磁铁在 Link 6 前方 8cm
            
            link6_pos = transform_from_gripper_to_link6(gripper_pos, gripper_orient, translation)
            adjusted_quaternion = adjust_rotation(gripper_orient, -90)
            self.rmpflow.set_end_effector_target(target_position=link6_pos, target_orientation=adjusted_quaternion)
            actions = self.articulation_rmpflow.get_next_articulation_action()
 
            self.articulation.apply_action(actions) 
            # print(self.articulation.get_joint_positions())
            current_mag_position = self.mag.get_world_pose()[0]*1000       
            current_agent_positon = self.agent.get_world_pose()[0]*1000

            
            current_agent_vel_xy = self.calculate_xy_vel(current_mag_position, current_agent_positon)
  
            self.agent.set_linear_velocity(current_agent_vel_xy)

            self.simpleStep()
        
        return success



    def step(self, action):
        self.step_num += 1
        # if self.step_num % self.skip_frame == 0
        action_success = self.take_discrete_action(action)
        current_agent_local_postion, current_agent_world_postion, current_mag_position, \
            cam_hand_img, cam_1_img, cam_2_img = self.get_observations()
        
        done, trunc, reward, info = self.calculate_reward\
            (current_agent_local_postion, current_agent_world_postion, current_mag_position, action_success)
        
        observations = {
            'image': cam_hand_img,  # 实际图像数据
  
        }


        self.simpleStep()

        info = {}
    
        return observations, done, reward, trunc 
    


    def get_observations(self):
        cam_hand_img = self.cam_hand.get_rgb()
        cam_1_img = self.cam_01.get_rgb()
        cam_2_img = self.cam_02.get_rgb()
        current_agent_world_postion = self.agent.get_world_pose()[0]
        current_agent_local_postion = self.agent.get_local_pose()[0]
        current_mag_position = self.mag.get_world_pose()[0]

        return current_agent_local_postion, current_agent_world_postion, \
            current_mag_position, cam_hand_img, cam_1_img, cam_2_img

    def calculate_reward(self, current_agent_local_postion, \
                         current_agent_world_postion, current_mag_position,action_success):
        trunc = False
        done = False
        info = {}
        #每多一步都要扣一分
        reward = -1

        trunc = self.check_position_update(current_agent_local_postion)
        des = np.array([87,-88,0])
        # print(des)

        current_agent_local_postion[2] = 0
      

        norm_endpoint = np.linalg.norm(current_agent_local_postion - des)
        grid_x = round(current_agent_local_postion[0] / self.grid_size) * self.grid_size
        grid_y = round(current_agent_local_postion[1] / self.grid_size) * self.grid_size
        grid_position = (grid_x, grid_y)

        if grid_position not in self.explored_areas:
            self.explored_areas.add(grid_position)
            reward += 10  # 探索到新位置时增加的奖励
        else:
            reward -= 10

        if not action_success:
            reward = reward - 100

        if norm_endpoint < 1e-4:
            done = True
            reward = reward + 1000
            print("Done")
        
        if trunc:
            reward = reward - 1000
            print('MAG LOST AGENT')
        
        return done, trunc, reward, info

    def reset(self):
        self.homing()

        self.position_history = []  # 存储位置的历史记录
        self.no_change_count = 0  # 计数器
        self.explored_areas.clear()
        self.target.set_world_pose(self.target_home_pose[0])
        self.agent.set_linear_velocity(np.array([0,0,0]))
        self.agent.set_world_pose(self.agent_home_pose[0])
        current_agent_local_postion, current_agent_world_postion, current_mag_position, \
            cam_hand_img, cam_1_img, cam_2_img = self.get_observations()
        observations = {
            'image': cam_hand_img,  # 实际图像数据
            'done': False,  # 任务是否完成
            'truncated': False,  # 任务是否被截断
        }
        return observations
        # Implement the reset logic
        # ...

    def render(self, mode="human"):
        self.my_world.render()
        pass
        # Implement the rendering logic
        # ...

    def close(self):
        self._simulation_app.close()

    def seed(self, seed=None):
        pass
        # Implement the seeding logic
        # ...

    def simpleStep(self):
        # self._simulation_app.update()
        self.my_world.step()
    
    def homing(self, atol_position=1e-2, atol_orientation=5):
        from omni.isaac.core.utils.types import ArticulationAction
        """
        将机器人移动到预定的位置和姿态。

        :param target_position: 目标位置，格式为 [x, y, z]
        :param target_orientation: 目标欧拉角，格式为 [x, y, z] (度)
        :param translation: 电磁铁相对于机械臂末端的平移向量
        :param atol_position: 位置的接受误差
        :param atol_orientation: 姿态的接受误差
        :return: 是否成功到达目标位置和姿态
        """

        target_joint_angles = [0,1.62,-0.781,0.7327,0,0]
        robot_action = ArticulationAction(
            joint_positions=np.array(target_joint_angles),
        )
        tolerance = 0.01
       
        while True:
            self.articulation.apply_action(robot_action)

            current_joint_angles = self.articulation.get_joint_positions()
            self.simpleStep()

            if np.allclose(current_joint_angles, target_joint_angles, atol=tolerance):
                return True



if __name__ == "__main__":
    env = MazeEnv(headless=False)
    env.reset()
    t = 0
    while t <150:
       
       observations, done, reward, trunc = env.step(action=0)
       t = t+1

    env.reset()
    # env._simulation_app.update()
    t = 0
    while t <150:
       
       observations, done, reward, trunc = env.step(action=0)
       t = t+1


