#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA

#intializing multibodysystem
dp = ob.MultiBodySystem()

joint2 = ob.SphericalJoint()
joint1 = ob.SphericalJoint()

#intialziing such that pendulum is hanging to the right
joint2.q_init = SOA.quatfromrev(1*np.pi/2, "y")
joint1.q_init = SOA.quatfromrev(0, "y")

#defining link
link2 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.4]), joint=joint2)
link1 = ob.Link(mass=20.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)

#adding link (first link is added to the base, second link is added to the first link and so on)
dp.add_link(link2)
dp.add_link(link1)
dp.add_link(link2)
dp.add_link(link1)

#parameters for simulation
dt = 0.005
end_time = 10
tspan = np.arange(0, end_time + dt/2, dt) # dt/2 to include end_time

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravitcompute_pos_iny in z

dp.plot_initial_state("open")

# dp.simulate(tspan,V_base,A_base,"open")

# dp.plot_gen_velocities(savefig=False)

# dp.animation(config="open",step=5)