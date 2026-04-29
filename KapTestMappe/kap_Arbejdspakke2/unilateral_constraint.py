import numpy as np
from utils import objects as ob
from utils import soa as SOA

#setting up robot
robot = ob.MultiBodySystem()

#setting up joints
joint1 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()

#adding initial configurations
joint1.q_init = SOA.quatfromrev(np.pi/2,"y")

#setting up link
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)

#adding link to system
robot.add_link(link1)
#robot.add_link(link2)

#parameters for simulation
tspan = np.arange(0,1,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81

#robot.plot_initial_state("open")

robot.simulate(tspan,V_base,A_base,config="unilateral_constraints")

robot.plot_gen_velocities() # <-------- HUSK DET ER I LOCAL FRAME.

robot.animation(config="open",step=10)
