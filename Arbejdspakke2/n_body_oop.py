#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint5 = ob.SphericalJoint()
joint4 = ob.SphericalJoint()
joint3 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#intialziing such that pendulum is hanging to the right
joint5.q_init = SOA.quatfromrev(3*np.pi/4, "y")
joint4.q_init = SOA.quatfromrev(0, "y")
joint3.q_init = SOA.quatfromrev(0, "y")
joint2.q_init = SOA.quatfromrev(0, "y")
joint1.q_init = SOA.quatfromrev(0, "y")

#defining link
link5 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint5)
link4 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint4)
link3 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link5)
robot.add_link(link4)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,10,2*5e-3)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

robot.plot_initial_state("open")

robot.simulate(tspan,V_base,A_base,"open")

# should be specified [body 1, body 2, ..., body n]
z0 = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
robot.calc_energies(z0)

path = "Arbejdspakke2/results"
file_name = "energies"
robot.CSV_creator(path, file_name, "tspan", "PE", "KE", "TE")

#robot.plot_gen_velocities()

robot.animation(config="open",step=1)
