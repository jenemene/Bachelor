import numpy as np
from utils import soa as SOA
from utils import plotting as SOAplt
import time
from initial_configs import initial_configs_open as iniconf

def N_body_pendulum(n, link, tspan, state0):
    start = time.perf_counter()
    print("--- Simulation started ---")
    print(f"Progress (until {tspan[-1]:.0f}):")

    def ODEfun(t, state, n, link):
        #solve_ivp passes state as np.array. It is unpacked, and then passed to ATBI as a a list of form state = [theta,beta].

        #RBT is constant
        RBT = SOA.RBT(l_hinge)

        #unpacking state
        theta = state[:4*n]
        beta = state[4*n:]

        #normalizing quartenions
        SOA.normalize_quaternions(theta)
        
        #calculating theta_dot based on the derrivmap function
        theta_dot = np.zeros(len(theta))
        for i in range(n):
            idxq = 4*i #these indexes assume that we ONLY have spherical joints
            idxw = 3*i
            theta_dot[idxq:idxq+4] = SOA.derrivmap(theta[idxq:idxq+4],beta[idxw:idxw+3],"spherical")
            
        #Calculationg of generalized accelerations (beta_dot) - this requires ATBI. 
        tau_vec = np.zeros_like(beta) #no external torques

        A, V, beta_dot_list, *unused = SOA.ATBI(state,tau_vec,n,link,t)

        beta_dot = np.concatenate([b.flatten() for b in beta_dot_list[1:n+1]])

        state_dot = np.concatenate([theta_dot, beta_dot.flatten()])

        # 1. Initialize 't_old' only on the very first call
        if not hasattr(ODEfun, "t_old"):
            ODEfun.t_old = -1 

        t_now = int(t)

        # 2. Check if the integer part of time has increased
        if t_now > ODEfun.t_old:
            print(t_now)
            ODEfun.t_old = t_now # Update memory to current second

        return state_dot, V

    Y, V_values = SOA.RK4_int_with_V(ODEfun, state0, tspan, n, link)

    end = time.perf_counter()
    print(f"Integration time: {end - start:.2f} seconds")
    print("--- Simulation finished ---")
    return Y, V_values, link

### LINK SETUP ###
m = 20
l_hinge = np.array([0,0,0.2])
link = SOA.SimpleLink(m, l_hinge)
link.set_hingemap("spherical")

### SIMULATION SETTINGS ###
n_bodies = 20
simulation_length = 10
dt = 0.001
state0 = iniconf.N_vertical(n_bodies)

### RUN SIMULATION ###
tspan = np.arange(0, simulation_length+dt, dt)
Y, V_values, link = N_body_pendulum(n_bodies, link, tspan, state0)

### ANIMATION ###
anim = SOAplt.animation_plot_moving_base(Y, tspan, link, "open", step=10)

### ENERGY CHECK ###
#SOAplt.check_energies(Y, V_values, tspan, link, n_bodies, "open", TE_only=True)