#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints

joint3 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right

joint3.q_init = SOA.quatfromrev(0.5*np.pi, "y")
#defining link
link3 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 1]), joint=joint3)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 1]), joint=joint2)
link1 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint1)


#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3)
robot.add_link(link2)

#parameters for simulation
tspan = np.arange(0,10,0.0001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z


robot.plot_initial_state("open")


robot.simulate(tspan,V_base,A_base,"open",BG_params=[100,500])


robot.plot_gen_velocities()
robot.plot_gen_accelerations(config="open", V_base=None, A_base=A_base, BG_params=None)

robot.animation(config="open",step=1)
robot.check

# -- hvis man vil have den lidt ude (husk da også at lav et +0.1 i constrainten selv) -- 
#joint3 = ob.FreeJoint()
#joint3.q_init = np.hstack([SOA.quatfromrev(0.5*np.pi, "y"),np.array([0.1,0,0])]) 