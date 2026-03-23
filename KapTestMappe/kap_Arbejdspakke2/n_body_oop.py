#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint2 = ob.RevoluteJoint(axis="y")
joint1 = ob.RevoluteJoint(axis="y")

#initial conditions
joint2.q_init = np.array([np.pi])
joint1.q_init = np.array([3*np.pi/2])

#defining link
link2 = ob.Link(mass=2.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=2.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,5,0.01)

V_base = np.zeros(6)
A_base = np.zeros(6)
#A_base[-1] = 9.81 #simulating gravity in z

robot.plot_initial_state(config="open")

BG = np.array([0,0])
robot.simulate(tspan, V_base, A_base, config="driver_bottom", BG_params=BG)

#robot.plot_gen_velocities()

robot.animation(config="open",step=50)
