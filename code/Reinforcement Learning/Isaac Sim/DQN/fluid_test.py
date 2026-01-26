from omni.isaac.kit import SimulationApp

CONFIG = {"renderer": "RayTracedLighting", "headless": False}
simulation_app = SimulationApp(CONFIG)

from omni.isaac.core import SimulationContext, utils  # noqa E402
from omni.isaac.core import World, PhysicsContext
from omni.isaac.core.utils.stage import add_reference_to_stage
import omni
from omni.isaac.core.utils import carb
from omni.isaac.core.utils.stage import open_stage
physics_dt=1.0 / 60.0
rendering_dt=1.0 / 60.0



# Loading the cube with the particle sampler as a reference
open_stage("/home/jialin/Documents/isaacFiles/fluid_test.usda")
#instanciate the world
world = World(
    # physics_prim_path = env_config["env"]["world"]["physics_prim_path"],
    physics_dt = physics_dt,
    rendering_dt = rendering_dt,

    device = "cuda",
    backend = "torch",
    set_defaults=True
    )
world.reset()
    # set up the physics context
# omni.timeline.get_timeline_interface().play()
world.play()
# physics_context = PhysicsContext(
#                         prim_path='/physicsScene',
#                         # prim_path = env_config["env"]["world"]["physics_prim_path"],
#                         device = "cuda",
#                         backend = "torch",
                        
#                         )


# physics_context.enable_gpu_dynamics(True)

# print(physics_context.use_gpu_pipeline)

# # if physics_context.use_gpu_pipeline:
# #     physics_context.enable_flatcache(True)

# # carb.set_carb_setting(world._settings, "/persistent/omnihydra/useSceneGraphInstancing", True)
# physics_context.enable_fabric(enable=True) 

# # world.scene.add_ground_plane()



# # simulation_app.update()
# # world.initialize_physics()

# # 

# while simulation_app.is_running():
#     world.step(render=True)
    
# # physics_context.enable_fabric()    physics_context._step()

# world.stop()
# simulation_app.close()