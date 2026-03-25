#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint3 = ob.FreeJoint()

#initial conditions
quat = SOA.quatfromrev(np.pi, "y")
pos = np.array([0, 0, 0])
joint3.q_init = np.concatenate([quat, pos])

#defining link
link3 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint3)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3)

#parameters for simulation
tspan = np.arange(0,2,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
#A_base[-1] = -9.81 #simulating gravity in z

#robot.plot_initial_state(config="open")

BG = np.array([0,0])
robot.simulate(tspan, V_base, A_base, config="driver", BG_params=BG)

#robot.plot_gen_velocities()

robot.animation(config="open",step=30)
