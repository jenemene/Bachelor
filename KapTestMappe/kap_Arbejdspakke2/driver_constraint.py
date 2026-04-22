import numpy as np
from utils import objects as ob
from utils import soa as SOA

#setting up robot
robot = ob.MultiBodySystem()

#setting up joints
joint5 = ob.FreeJoint()
joint4 = ob.SphericalJoint()
joint3 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#adding initial configurations for the base join (make it hang down plus make it adhere to driver)
quat = SOA.quatfromrev(np.pi+np.pi/6,"y")
pos = np.array([0.2,0,0])
joint5.q_init = np.concatenate([quat,pos])
joint4.q_init = SOA.quatfromrev(-np.pi/6,"y")
joint3.q_init = SOA.quatfromrev(3*np.pi/2,"y")
joint2.q_init = SOA.quatfromrev(3*np.pi/2,"y")
joint1.q_init = SOA.quatfromrev(np.pi/6,"y")

#setting up link
link5 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint5)
link4 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint4)
link3 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link to system
robot.add_link(link5)
robot.add_link(link4)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,0.895,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 

robot.plot_initial_state("closed")

robot.simulate(tspan,V_base,A_base,config="driver",BG_params=[100,500])

robot.plot_gen_velocities() #<-------- HUSK DET ER I LOCAL FRAME. 

robot.animation(config="open",step=30)
