#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint7 = ob.SphericalJoint()
joint6 = ob.SphericalJoint()
joint5 = ob.SphericalJoint()
joint4 = ob.SphericalJoint()
joint3 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#initial conditions
joint7.q_init = SOA.quatfromrev(np.pi, "y")
joint6.q_init = SOA.quatfromrev(0, "y")
joint5.q_init = SOA.quatfromrev(0, "y")
joint4.q_init = SOA.quatfromrev(3*np.pi/2, "y")
joint3.q_init = SOA.quatfromrev(3*np.pi/2, "y")
joint2.q_init = SOA.quatfromrev(0, "y")
joint1.q_init = SOA.quatfromrev(0, "y")

#defining link
link7 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint7)
link6 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint6)
link5 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint5)
link4 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint4)
link3 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint3)
link2 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link7)
robot.add_link(link6)
robot.add_link(link5)
robot.add_link(link4)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,5,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravity in z

robot.plot_initial_state(config="closed")

BG = np.array([50,50])
robot.simulate(tspan, V_base, A_base, config="driver_bottom", BG_params=BG)

#robot.plot_gen_velocities()

robot.animation(config="closed",step=30)
