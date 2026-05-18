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

#defining link
link3_fixed= ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3_fixed)
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)


#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3_fixed)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 100
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

# robot.plot_initial_state("closed")

robot.simulate(tspan,V_base,A_base,"closed",BG_params=[0,500])

robot.calc_TE_delta()

path = "Arbejdspakke2/results"
file_name = "3_closed_energies_t100"
robot.CSV_creator(path, file_name, "tspan", "TE_delta")

file_name = "3_closed_gen_acc_t100"
robot.CSV_creator(path, file_name, "tspan", "beta_dot")

# robot.plot_gen_velocities(savefig=False)

# robot.animation(config="closed",step=5)