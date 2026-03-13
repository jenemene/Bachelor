#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint1 = ob.RevoluteJoint(axis="y")
joint2 = ob.RevoluteJoint(axis="x")
joint3 = ob.RevoluteJoint(axis="y")

#intialziing such that pendulum is hanging to the right
joint3.q_init = np.array([np.pi/2])
joint2.q_init = np.array([np.pi/2])
joint1.q_init = np.array([np.pi/2])

#defining link
link1 = ob.Link(mass=2.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link3 = ob.Link(mass=0.5, l_hinge=np.array([0, 0, 0.2]), joint=joint3)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link1)
robot.add_link(link2)
robot.add_link(link3)

#parameters for simulation
tspan = np.arange(0,5,0.01)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravity in z

robot.plot_initial_state(config="open")

#robot.simulate_open(tspan,V_base,A_base)

#robot.plot_gen_velocities()

#robot.animation(config="open",step=30)
