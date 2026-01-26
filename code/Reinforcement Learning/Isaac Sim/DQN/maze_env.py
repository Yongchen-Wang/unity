import gym
from gym import spaces
import numpy as np
import math
import carb
import torch
import matplotlib.pyplot as plt
# from omni.isaac.examples.base_sample import BaseSample

# Can be used to create a new agent or to point to an already existing agent in stage.



class MazeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}


    def __init__(
        self,
        skip_frame=3,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=256,
        seed=0,
        headless=True,
        
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core.prims import RigidPrimView
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.sensor import ContactSensor, Camera
        
        self.my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0, device = "cuda",backend = "torch",)
        self.my_world.scene.add_ground_plane()
        self.maze_asset_path = "/home/jialin/Documents/isaacFiles/myMaze.usd"

        add_reference_to_stage(usd_path=self.maze_asset_path, prim_path="/World/myMaze")

        self.my_world.play()
        self.agent = RigidPrimView(prim_paths_expr="/World/myMaze/The_Maze/agent/Sphere")
        self.agent.initialize()
        self.camera = Camera(prim_path="/World/myMaze/The_Maze/Camera")
        self.camera.initialize()


        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=float("inf"), high=float("inf"), shape=(8,), dtype=np.float32)
        self.max_velocity = 1
        self.reset_counter = 0
        self.goal_position = np.array([-3.5,3.5])
        self.visited_positions = []
        
        self.camera.add_motion_vectors_to_frame()
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        # previous_agent_position, _ = self.agent.get_world_pose()
        v_x = action[0]
        v_y = action[1]

        for i in range(self._skip_frame):
            velocity_tensor = torch.tensor([v_x, v_y, 0.0, 0.0,0.0,0.0], dtype=torch.float32)
            self.agent.set_velocities(velocity_tensor)
            self._my_world.step(render=False)
        
        agent_world_position, agent_linear_velocity, collision = self.get_observations()
        agent_world_position = agent_world_position.cpu().numpy() if isinstance(agent_world_position, torch.Tensor) and agent_world_position.is_cuda else agent_world_position
        agent_linear_velocity = agent_linear_velocity.cpu().numpy() if isinstance(agent_linear_velocity, torch.Tensor) and agent_linear_velocity.is_cuda else agent_linear_velocity
        collision = collision.cpu().numpy() if isinstance(collision, torch.Tensor) and collision.is_cuda else collision
        self.goal_position = self.goal_position.cpu().numpy() if isinstance(self.goal_position, torch.Tensor) and self.goal_position.is_cuda else self.goal_position

        agent_world_position_xy = agent_world_position[:2]
        agent_linear_velocity_xy = agent_linear_velocity[:2]
        print(agent_world_position_xy)

        info = {}
        done = False
        reward, done = self.calculate_reward(agent_world_position_xy,collision[0])
        observations = np.concatenate(
            [
                agent_world_position_xy,
                agent_linear_velocity_xy,
                collision,
                self.goal_position,
            ]
        )
        return observations, reward, done, info


    def calculate_reward(self, agent_position, collision_status):
        # Define reward factors
        goal_reached_reward = 1000.0  # Reward for reaching the goal
        collision_penalty = -5  # Penalty for collision
        time_penalty = -1  # Penalty for each time step taken

        exploration_reward = 5
        reward = 0
        # Calculate Euclidean distance to the goal
        distance_to_goal = np.linalg.norm(self.goal_position - agent_position)

        reward = -distance_to_goal*10
        # Apply collision penalty if a collision occurred
        if collision_status:
            reward = reward - collision_penalty

            # Reward is inversely proportional to the distance to the goal
             # Scale the distance influence\
        
        
        # if not self.has_been_visited(agent_position):
        #     reward += exploration_reward
        #     self.mark_as_visited(agent_position)

        # print(len(self.visited_positions))
       

        # Check if the goal has been reached
        if distance_to_goal < 0.5:  # Threshold for reaching the goal
            reward += goal_reached_reward
            done = True
        else:
            done = False

        # Apply time penalty for taking steps
        # reward += time_penalty

        return reward, done


                    

        
    def get_observations(self):
            self._my_world.render()
            imgplot = plt.imshow(self.camera.get_rgba()[:, :, :3])
            plt.show()

            camdict = self.camera.get_current_frame()
            print(camdict)
            agent_world_position = self.agent.get_world_pose()
            agent_linear_velocity = self.agent.get_linear_velocity()
            collision = 1.0 if self.sensor.get_current_frame()['force'] > 0.2 else 0.0
            # goal_world_position, _ = self.goal.get_world_pose()
            return agent_world_position[0], agent_linear_velocity, np.array([collision,0])

    def reset(self):
        self._my_world.reset()
        self.reset_counter = 0
        self.agent.set_default_state(position=np.array([3.5, -3.5, 0.25]), linear_velocity=np.array([0, 0, 0]))
        # randomize goal location in circle around robot
        self.visited_positions.clear()
        agent_world_position, agent_linear_velocity, collision = self.get_observations()

        agent_world_position = agent_world_position.cpu().numpy() if isinstance(agent_world_position, torch.Tensor) and agent_world_position.is_cuda else agent_world_position
        agent_linear_velocity = agent_linear_velocity.cpu().numpy() if isinstance(agent_linear_velocity, torch.Tensor) and agent_linear_velocity.is_cuda else agent_linear_velocity
        collision = collision.cpu().numpy() if isinstance(collision, torch.Tensor) and collision.is_cuda else collision
        self.goal_position = self.goal_position.cpu().numpy() if isinstance(self.goal_position, torch.Tensor) and self.goal_position.is_cuda else self.goal_position

        
        agent_world_position_xy = agent_world_position[:2]
        agent_linear_velocity_xy = agent_linear_velocity[:2]
        observations = np.concatenate(
            [
                agent_world_position_xy,
                agent_linear_velocity_xy,
                collision,
                self.goal_position,
            ]
        )
        return observations

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]


    def simpleStep(self):
        self._my_world.step(render=False)