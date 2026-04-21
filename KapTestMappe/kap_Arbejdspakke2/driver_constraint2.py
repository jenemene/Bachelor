import numpy as np
from utils import objects as ob
from utils import soa as SOA

#setting up robot
robot = ob.MultiBodySystem()

#setting up joints
joint5 = ob.FreeJoint()
joint4 = ob.RevoluteJoint("y")

#adding initial configurations for the base join (make it hang down plus make it adhere to driver)
quat = SOA.quatfromrev(np.pi+np.pi/6,"y")
pos = np.array([0.2,0,0])
joint5.q_init = np.concatenate([quat,pos])


#setting up link
link5 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint5)
link4 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint4)

#adding link to system
robot.add_link(link5)
robot.add_link(link4)

#parameters for simulation
tspan = np.arange(0,1.22,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 

robot.plot_initial_state("closed")

robot.simulate(tspan,V_base,A_base,config="driver2",BG_params=[100,500])

robot.plot_gen_velocities() #<-------- HUSK DET ER I LOCAL FRAME. 

robot.animation(config="open",step=1)
