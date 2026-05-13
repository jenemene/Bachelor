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

joint3.q_init = SOA.quatfromrev(3*np.pi/2,"y")
joint2.q_init = SOA.quatfromrev(0,"y")
joint1.q_init = SOA.quatfromrev(0,"y")

#defining link
link3 = ob.Link(mass=1.0, l_hinge=np.array([0.0, 0.0, 0.2]), joint=joint3)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0.0, 0.0, 0.2]), joint=joint2)
link1 = ob.Link(mass=1.0, l_hinge=np.array([0.0, 0.0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,2,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravity in z

robot.plot_initial_state("open")

robot.simulate(tspan,V_base,A_base,"open",BG_params=[0,0])

file_path_name = "KapTestMappe/kap_arbejdspakke2/results/test"
robot.CSV_creator(file_path_name, "tspan", "result")

robot.plot_gen_velocities()

robot.animation(config="open",step=30)
