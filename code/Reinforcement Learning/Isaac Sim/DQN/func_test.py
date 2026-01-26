import omni
import numpy as np
from omni.isaac.kit import SimulationApp

CONFIG = {"width": 1280, "height": 720, "sync_loads": True, "headless": False, "renderer": "RayTracedLighting"}
kit = SimulationApp(launch_config=CONFIG)
maze_asset_path = "/home/jialin/Documents/isaacFiles/magtest.usd"
omni.usd.get_context().open_stage(maze_asset_path)

kit.update()
kit.update()


print("Loading stage...")
from omni.isaac.core.utils.stage import is_stage_loading
from omni.isaac.core import World
from omni.isaac.motion_generation.lula import RmpFlow
from omni.isaac.core.articulations import Articulation
from omni.isaac.motion_generation import ArticulationMotionPolicy
from omni.isaac.core.objects import cuboid
from omni.isaac.core.prims import RigidPrim,GeometryPrim,RigidPrimView
while is_stage_loading():
    kit.update()
print("Loading Complete")


# articulation = Articulation(prim_path="/UnitreeZ1")
       
    


# omni.timeline.get_timeline_interface().play()
# my_world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0, device = "cuda",backend = "torch",set_defaults=False)
my_world = World()
my_world.play()
# my_world.reset()
# articulation.initialize()

# rmpflow = RmpFlow(
#             robot_description_path = "/home/jialin/Documents/isaacFiles/z1/newLula.yaml",
#             urdf_path ="/home/jialin/Documents/isaacFiles/z1/z1.urdf",
#             rmpflow_config_path ="/home/jialin/Documents/isaacFiles/z1/rmp_config_z1.yaml",
#             end_effector_frame_name = "link06",
#             maximum_substep_size = 0.00334
#         )
# articulation_rmpflow = ArticulationMotionPolicy(robot_articulation=articulation,motion_policy=rmpflow,default_physics_dt=1/60)

# target_cube = cuboid.VisualCuboid("/World/target",position=np.array([0.,0,.1]),color=np.array([1.,0,0]),size=.1)
# mag = RigidPrim(prim_path="/UnitreeZ1/link06/magnet")

import magpylib as magpy
from numpy import linspace, pi
from matplotlib.pyplot import show
import numpy as np
from magpylib.magnet import Cylinder
from magforce import getM, getF



mag = GeometryPrim(prim_path="/World/Cylinder")
target = RigidPrim("/World/Sphere")
current_mag_position = mag.get_world_pose()[0]*1000
current_target_position = target.get_world_pose()[0]*1000


mag_sensor = magpy.Sensor(position=current_target_position)
last_target_position = current_target_position
last_mag_position = current_mag_position
demagnetizing_factor = 1/3             # sphere
volume = 4 / 3 * pi * (7.5) ** 3  # V sphere r=4mm [m3]
M_saturation = 480000               # Ms Co room temperature [A/m]

sample = {'demagnetizing_factor': demagnetizing_factor,
          'volume': volume,
          'M_saturation': M_saturation}


target.set_density(7900)

radius = 7.5 / 1000  # 半径转换为米
density = 7900  # 密度 (kg/m³)
mass = (4/3) * pi * (radius ** 3) * density  # 计算质量

# 时间步长 (假设)
delta_t = 1/60  # 10毫秒，根据您的模拟设置调整

# 初始化速度
velocity = np.array([0.0, 0.0, 0.0])


while kit.is_running():
    

    current_mag_position = mag.get_world_pose()[0]*1000

    m1 = Cylinder(magnetization=[0, 0, -1000], dimension=(20,15),
              position= current_mag_position, orientation=None) 
    
    current_target_position = target.get_world_pose()[0]*1000


 
    # 计算作用力
    F_000_m1 = getF(current_target_position, m1, sample)
    F_total = np.array([F_000_m1[0],F_000_m1[1],0])

   
    # 计算瞬时速度而不是加速度
    # 注意：这里假设 delta_t 是足够小的，以至于速度变化不会太剧烈
    velocity = F_total / mass * delta_t

    print(velocity)



    # 设置小球的线速度
    target.set_linear_velocity(velocity)

        
  
    # rmpflow.set_end_effector_target(target_position=(target.get_world_pose()[0]))
    # actions = articulation_rmpflow.get_next_articulation_action()
    # # print(actions.get_dict())
    
    # articulation.apply_action(actions)
        # Run in realtime mode, we don't specify the step size
    # my_world.step()
    kit.update()

# kit.stop()
kit.close()
