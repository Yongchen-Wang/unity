import magpylib as magpy
from numpy import linspace, pi
from matplotlib.pyplot import show
import numpy as np
from magpylib.magnet import Cylinder
from magforce import getM, getF

target_position = np.array([ 0, 0.0,  200], dtype=float)
m1 = Cylinder(magnetization=[0, 0, -1000], dimension=(20,15),
              position= target_position, orientation=None)  

# current mag pos is : [1.9990088e+02 1.3350800e-03 1.4985316e+02]
# delat_ mag _pos is [ 1.5258789e-05 -1.6356353e-05  4.5776367e-05]
# sample position is : [201.5028     -1.3711157  74.59483  ]
# [-1730.08834891  1689.07945392 75167.48097408]

demagnetizing_factor = 1/3             # sphere
volume = 4 / 3 * pi * (4) ** 3  # V sphere r=4mm [m3]
M_saturation = 480000               # Ms Co room temperature [A/m]

sample = {'demagnetizing_factor': demagnetizing_factor,
          'volume': volume,
          'M_saturation': M_saturation}


point = np.array([0, 0, 0])/1000

F_000_m1 = getF(point, m1, sample)
sensor = magpy.Sensor()

print(f'F_000_m1 = {F_000_m1}')
# magpy.show(m1, sensor, backend='plotly')
# Create a Cuboid magnet with magnetic polarization
# of 1000 mT pointing in x-direction and sides of
# 1,2 and 3 mm respectively.

# cube = magpy.magnet.Cuboid(magnetization=(1000,0,0), dimension=(1,2,3))

# # Create a Sensor for measuring the field

# sensor = magpy.Sensor()

# # By default, the position of a Magpylib object is
# # (0,0,0) and its orientation is the unit rotation,
# # given by a scipy rotation object.

# print(cube.position)                   # -> [0. 0. 0.]
# print(cube.orientation.as_rotvec())    # -> [0. 0. 0.]

# # Manipulate object position and orientation through
# # the respective attributes:

# from scipy.spatial.transform import Rotation as R
# cube.position = (1,0,0)
# cube.orientation = R.from_rotvec((0,0,45), degrees=True)

# print(cube.position)                            # -> [1. 0.  0.]
# print(cube.orientation.as_rotvec(degrees=True)) # -> [0. 0. 45.]

# # Apply relative motion with the powerful `move`
# # and `rotate` methods.
# sensor.move((-1,0,0))
# sensor.rotate_from_angax(angle=-45, axis='z')

# print(sensor.position)                            # -> [-1.  0.  0.]
# print(sensor.orientation.as_rotvec(degrees=True)) # -> [ 0.  0. -45.]