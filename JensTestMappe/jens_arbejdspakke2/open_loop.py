#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint_free = ob.FreeJoint()
joint_trans = ob.TranslationalJoint()
joint_rot = ob.SphericalJoint()
joint3 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right
joint_trans.q_init = np.array([0.0, 0.0, 0.0])
joint_rot.q_init = SOA.quatfromrev(0.5 * np.pi, "y")


joint3.q_init = SOA.quatfromrev(3/4*np.pi, "y")

joint_free.q_init = np.hstack([SOA.quatfromrev(np.pi, "y"),np.array([0,0,0])])


joint_free.w_init = np.array([0,1,0,0,0,0])



#defining link
link_free = ob.Link(mass=100, l_hinge=np.array([0.0, 0.0, 1.0]), joint=joint_free)
link3 = ob.Link(mass=100.0, l_hinge=np.array([0, 0, 1]), joint=joint3)
link2 = ob.Link(mass=100.0, l_hinge=np.array([0, 0, 1]), joint=joint2)
link1 = ob.Link(mass=100.0, l_hinge=np.array([0, 0, 1]), joint=joint1)


#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link_free)
#robot.add_link(link3)
#robot.add_link(link2)
#robot.add_link(link1)





#parameters for simulation
tspan = np.arange(0,10,0.005)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 0*9.81 #simulating gravitcompute_pos_iny in z


robot.plot_initial_state("open")

robot.simulate(tspan,V_base,A_base,"open")


robot.plot_gen_velocities()


robot.calc_TE_error()

path = "JensTestMappe/jens_arbejdspakke2/results"
file_name = "energy_free_hinge_open"
robot.CSV_creator(path, file_name, "tspan", "TE_error")


robot.animation(config="open",step=5)
