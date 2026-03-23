#imports
import numpy as np
from utils import objects as ob
from utils import soa as SOA
import time

#intializing multibodysystem
robot = ob.MultiBodySystem()

#defining joints
joint1 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint3 = ob.SphericalJoint()


#intialziing such that pendulum is hanging to the right
joint3.q_init = SOA.quatfromrev(1*np.pi, "y")
joint2.q_init = SOA.quatfromrev(2*np.pi/3, "y")
joint1.q_init = SOA.quatfromrev(2*np.pi/3, "y")

#defining link
link1 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint1)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint2)
link3 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, 0.2]), joint=joint3)

#adding link (first link is added to the base, second link is added to the first link and so on)
robot.add_link(link3)
robot.add_link(link2)
robot.add_link(link1)

#parameters for simulation
tspan = np.arange(0,5,0.001)

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81 #simulating gravity in z

start = time.perf_counter()
robot.simulate(tspan,V_base,A_base,"closed",BG_params=[2000,2500])
end = time.perf_counter()
print(end-start)

#robot.plot_gen_velocities()

#robot.animation(config="closed",step=60)
