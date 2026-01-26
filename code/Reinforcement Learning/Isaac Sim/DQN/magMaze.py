import gym
from gym import spaces
import numpy as np


# from magpylib.magnet import Cylinder
from magforce import getM, getF
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
        resolution = 256
        
    ) -> None:
        from omni.isaac.kit import SimulationApp
        
        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        self.resolution = resolution 

    
        from omni.isaac.core import World
        from omni.isaac.core.prims import RigidPrim,GeometryPrim
        from omni.isaac.core.utils.stage import add_reference_to_stage
        from omni.isaac.sensor import Camera
        import omni
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.motion_generation import ArticulationMotionPolicy
        from omni.isaac.motion_generation.lula import RmpFlow
        from omni.isaac.core.objects import cuboid

        maze_asset_path = "/home/jialin/Documents/isaacFiles/simRLenv.usda"
        omni.usd.get_context().open_stage(maze_asset_path)

        self._simulation_app.update()
        self._simulation_app.update()

        print("Loading stage...")
        from omni.isaac.core.utils.stage import is_stage_loading
        # from omni.isaac.core import World
        while is_stage_loading():
            self._simulation_app.update()
        print("Loading Complete")
     
        # self.target_cube = cuboid.VisualCuboid("/World/target",position=np.array([0.,0,.5]),color=np.array([1.,0,0]),size=.1)
        # self.my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0, device = "cuda",backend = "torch",set_defaults=False)
        self.my_world = World()
        # self.my_world.scene.add_ground_plane()

        # self.stage = Usd.Stage.Open(self.maze_asset_path)

        # self.my_world.stage.Open(self.maze_asset_path)
        # add_reference_to_stage(usd_path=self.maze_asset_path, prim_path=self.maze_asset_path)
        # self.my_world = World()
        # self.my_world.stage.Open(self.maze_asset_path)
        self.agent = RigidPrim(prim_path="/maze3/agent")
        # self.agent = RigidPrimView(prim_paths_expr="/World/myMaze/The_Maze/agent/Sphere")
                   
        self.mag = RigidPrim(prim_path="/UnitreeZ1/link06/magnet")

        # self.camera = Camera(prim_path="/World/myMaze/The_Maze/Camera", resolution=(self.resolution,self.resolution))



        # self.mag_cal_obj = Cylinder(magnetization=[0, 0, 1300], dimension=(100,100),
        #       position= [350, -350, 250])   # center is at z = -20mm
        
        # demagnetizing_factor = 1/3             # sphere
        # volume = 4 / 3 * np.pi * (50/ 1000) ** 3  # V sphere r=4mm [m3]
        # M_saturation = 1.400e6                 # Ms Co room temperature [A/m]

        # self.sample = {'demagnetizing_factor': demagnetizing_factor,
        #         'volume': volume,
        #         'M_saturation': M_saturation}

       
        
        # self.my_world.play()
        
        # self.my_world.scene.add(self.mag)

        # self.agent.initialize()
        # self.mag.initialize()
        # self.camera.initialize()
      
        # omni.timeline.get_timeline_interface().play()
        self.articulation = Articulation(prim_path="/UnitreeZ1")
        self.my_world.play()
        
        self.my_world.reset()
    
        self.articulation.initialize()

        self.rmpflow = RmpFlow(
            robot_description_path = "/home/jialin/Documents/isaacFiles/z1/newLula.yaml",
            urdf_path ="/home/jialin/Documents/isaacFiles/z1/z1.urdf",
            rmpflow_config_path ="/home/jialin/Documents/isaacFiles/z1/rmp_config_z1.yaml",
            end_effector_frame_name = "link06",
            maximum_substep_size = 0.00334
        )
        self.articulation_rmpflow = ArticulationMotionPolicy(robot_articulation=self.articulation,motion_policy=self.rmpflow,default_physics_dt=physics_dt)
        # self.target_cube = cuboid.VisualCuboid("/World/target",position=np.array([0.,0,.1]),color=np.array([1.,0,0]),size=.1)


        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-1, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.resolution, self.resolution, 3), dtype=np.uint8)
  
        self.reset_counter = 0
        self.goal_position = np.array([-3.5,3.5])
        
        # self.camera.add_motion_vectors_to_frame()
        return

    def get_dt(self):
        return self._dt

    def step(self, action):
        # previous_agent_position = self.agent.get_world_poses()[0].cpu().numpy()[0]

        for i in range(self._skip_frame):
            # self.mag.set_angular_velocity()
            # self.mag.set_local_pose(translation=np.array([0.0, 0.1, 0.0]))
            # self.mag.set_linear_velocity(np.array([0.0, 0.01, 0.0]))

            # mag_position = self.mag.get_world_poses()[0]
             
            # self.mag_cal_obj.move
            # F = getF(previous_agent_position, self.mag_cal_obj, self.sample)
            # force = torch.tensor(np.array([[0,100000,0]]), dtype=torch.float)
            # self.agent.apply_forces(forces=force)
            # self.mag.apply_forces(forces=force)
            if self.articulation is None:
                print("articultaion is none")
                return
            if self.articulation_rmpflow is None:
                print("rmp is none")
                return
            self.rmpflow.set_end_effector_target(target_position=(self.target_cube.get_world_pose()[0]))
            actions = self.articulation_rmpflow.get_next_articulation_action()
            # print(actions.get_dict())
            
            self.articulation.apply_action(actions)
            self.simpleStep()

        info = {}
        done = False
   
        observations = self.get_observations()
        reward = 1
        return observations, reward, done, info



    
        
    def get_observations(self):
            # self.my_world.render()
            # img = self.camera.get_rgb()

            pose = self.mag.get_local_pose()
            # print(pose)
            observations = None
          
            
          
   
            return observations

    def reset(self):
        # self._simulation_app.
        # import omni
        # omni.timeline.get_timeline_interface().stop()
        # omni.timeline.get_timeline_interface().play()
        self.my_world.reset()
        self.reset_counter = 0
        observations = self.get_observations()
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
        self._simulation_app.update()

if __name__ == "__main__":
    env = MazeEnv(headless=False)
    env.reset()

    while True:
        env.step(action=1)
