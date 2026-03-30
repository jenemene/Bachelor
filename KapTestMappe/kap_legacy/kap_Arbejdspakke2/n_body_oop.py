#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint1 = ob.FreeJoint()

#intialziing such that pendulum is hanging to the right
quat = SOA.quatfromrev(0*np.pi/4,"y")
pos = [0,0,0]
joint1.q_init = np.concatenate([quat,pos])

#defining link
link1 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.02]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,2,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = -9.81 #simulating gravity in z

robot.plot_initial_state("open")

robot.simulate(tspan,V_base,A_base,"open",BG_params=[0,0])

robot.plot_gen_velocities()

robot.animation(config="open",step=30)
