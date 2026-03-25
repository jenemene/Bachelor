#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint1 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right
joint2.q_init = SOA.quatfromrev(np.pi/4, "y")
joint1.q_init = SOA.quatfromrev(np.pi/2, "y")

#defining link
link1 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint1)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint2)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,5,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravity in z


#robot.plot_initial_state("open")


robot.simulate(tspan,V_base,A_base,"driver_bottom",BG_params=[100,500])


#robot.plot_gen_velocities()

robot.animation(config="closed",step=60)
