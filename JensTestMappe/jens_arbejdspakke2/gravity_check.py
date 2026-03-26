#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint3 = ob.SphericalJoint()
joint2 = ob.FreeJoint()
joint1 = ob.SphericalJoint()

quat = SOA.quatfromrev(0.35*np.pi,"y")
pos = np.array([0.0,0,0])
joint2.q_init = np.concatenate([quat,pos])
joint1.q_init = SOA.quatfromrev(np.pi/2,"y")
joint3.q_init = SOA.quatfromrev(1.0*np.pi,"y")

#defining link
link2 = ob.Link(mass=1.0, l_hinge=np.array([0.0, 0.0, 0.2]), joint=joint2)
link1 = ob.Link(mass=5.0, l_hinge=np.array([0, 0, 1]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link2)

#parameters for simulation
tspan = np.arange(0,2,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravity in z


robot.plot_initial_state("open")


robot.simulate(tspan,V_base,A_base,"open",BG_params=[0,0])


robot.plot_gen_velocities()

robot.animation(config="open",step=60)
