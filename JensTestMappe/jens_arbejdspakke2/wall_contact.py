import numpy as np
from utils import objects as ob
from utils import soa as SOA

# 1. Initialize multibody system
robot = ob.MultiBodySystem()

# 2. Define Joints
joint1 = ob.SphericalJoint()
joint2 = ob.SphericalJoint()
joint3 = ob.SphericalJoint()
joint3FREE = ob.FreeJoint()

# 3. Initialization setup
# Link local vectors are [0, 0, L]. To make it hang straight down (-Z), 
# we rotate the base joint 180 degrees (pi) around the Y-axis.
joint1.q_init = SOA.quatfromrev(0.0, "y")
joint2.q_init = SOA.quatfromrev(0.0, "y")
joint3.q_init = SOA.quatfromrev(3/4*np.pi, "y")
joint3FREE.q_init = np.hstack([SOA.quatfromrev(2/4*np.pi, "y"),np.array([0.0,0,0])]) 


# The "Hammer Strike"
# We give the base joint an initial angular velocity of 6.0 rad/s around the Y-axis.
# Using the right-hand rule, this will violently swing the pendulum toward positive X!
joint1.w_init = np.array([0.0, 0.0, 0.0])
joint2.w_init = np.zeros(3)
joint3.w_init = np.zeros(3)

# 4. Define Links 
# Total length = 3 * sqrt(0.02) = 0.424m
# We place the wall at X = 0.35m in the solver, so it will definitely hit it.
L = np.sqrt(0.02)
L=0.5
link3FREE = ob.Link(mass=1.0, l_hinge=np.array([0, 0, L]), joint=joint3FREE)
link3 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, L]), joint=joint3)
link2 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, L]), joint=joint2)
link1 = ob.Link(mass=1.0, l_hinge=np.array([0, 0, L]), joint=joint1)

# 5. Add links (Remember your architecture adds them in reverse order!)
#robot.add_link(link3)
robot.add_link(link3FREE)
robot.add_link(link2)
robot.add_link(link1)
robot.add_link(link1)

# 6. Simulation Parameters
# 3 seconds is plenty of time to see it swing, hit the wall, and bounce back
tspan = np.arange(0, 3.0, 0.0001) 

V_base = np.zeros(6)
A_base = np.zeros(6)
A_base[-1] = 9.81  # Simulating gravity (accelerating the base upwards pushes links down)

# 7. Run and Plot
robot.plot_initial_state("open")

print("Running Unilateral Wall Contact Simulation...")
# High Baumgarte parameters [alpha, beta] act like a stiff, bouncy spring for the wall
robot.simulate(tspan, V_base, A_base, "wall_contact", BG_params=[200, 200])

print("Rendering Animation...")
# config="open" tells the animator to draw a standard straight chain (no closed loops)
robot.animation(config="open", step=20) # step=2 skips frames to make the animation play faster

# Uncomment this to see the velocity spike when it hits the wall!
robot.plot_gen_velocities()