#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
dp = ob.MultiBodySystem()

joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#intialziing such that pendulum is hanging to the right
joint2.q_init = SOA.quatfromrev(3*np.pi/4, "y")
joint1.q_init = SOA.quatfromrev(0, "y")

#defining link
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
dp.add_link(link2)
dp.add_link(link1)
dp.add_link(link1)
dp.add_link(link1)
dp.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 10
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

#dp.plot_initial_state("open")

dp.simulate(tspan,V_base,A_base,"open")

# should be specified [body 1, body 2, ..., body n]
z0 = np.array([0.9,0.7,0.5,0.3,0.1])
# z0 = np.array([0.3,0.1])
dp.calc_energies(z0)

path = "Arbejdspakke2/results"
file_name = "dp_energies"
dp.CSV_creator(path, file_name, "tspan", "PE", "KE", "TE")

file_name = "5p_gen_acc"
dp.CSV_creator(path, file_name, "tspan", "beta_dot")

#dp.plot_gen_velocities()

dp.animation(config="open",step=5)