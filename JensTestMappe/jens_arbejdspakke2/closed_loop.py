#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints

joint3_fixed = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right

joint3_fixed.q_init = SOA.quatfromrev(0.5*np.pi, "y")
joint2.q_init = SOA.quatfromrev(2*np.pi/3, "y")
joint1.q_init = SOA.quatfromrev(2*np.pi/3, "y")

joint3 = ob.FreeJoint()
joint3.q_init = np.hstack([SOA.quatfromrev(0.5*np.pi, "y"),np.array([0.0,0,0])]) 

#defining link
link3_fixed= ob.Link(mass=20.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint3_fixed)
link3 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint3)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, np.sqrt(0.02)]), joint=joint1)


#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3_fixed)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,10,0.01)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z


robot.plot_initial_state("closed")
 
# Calling with a specific solver and tighter tolerances
#robot.simulate_scipy(tspan, V_base, A_base, config="closed", BG_params=(0, 200), method='Radau', rtol=1e-6, atol=1e-9)

robot.simulate(tspan,V_base,A_base,"closed",BG_params=[0,250])

robot.calc_energies(np.array([np.cos(np.pi/6)*np.sqrt(0.02)/2,np.cos(np.pi/6)*np.sqrt(0.02),np.cos(np.pi/6)*np.sqrt(0.02)/2]))
#robot.plot_gen_velocities()
path = "JensTestMappe/jens_arbejdspakke2/results"
file_name = "CL_energies"
robot.CSV_creator(path, file_name, "tspan", "PE", "KE", "TE")
robot.animation(config="closed",step=1)
