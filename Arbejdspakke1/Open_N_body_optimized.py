import numpy as np
from utils import soa as SOA
from initial_configs import initial_configs_open as iniconf
import time
import pandas as pd

#N_body_pendulum set up

def N_body_pendulum(n, link, tspan, state0):

    def ODEfun(t, state, n, link):
        #solve_ivp passes state as np.array. It is unpacked, and then passed to ATBI as a a list of form state = [theta,beta].


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

        A, V, beta_dot_list, *unused = SOA.ATBI(state,tau_vec,n,link)

        beta_dot = np.concatenate([b.flatten() for b in beta_dot_list[1:n+1]])

        state_dot = np.concatenate([theta_dot, beta_dot.flatten()])

        return state_dot

    Y = SOA.RK4_int(ODEfun, state0, tspan, n, link)


    return Y

### LINK SETUP ###
m = 20
l_hinge = np.array([0,0,0.2])
link = SOA.SimpleLink(m, l_hinge)
link.set_hingemap("spherical")

### SIMULATION SETTINGS ###
n_list = np.arange(1,11,1) #number of bodies to simulate
simulation_length = 10
dt = 0.005
tspan = np.arange(0, simulation_length+dt, dt)


### S
repeats = 10
times = {} #a list nested with lists of time for each body
j = 0

for n_bodies in n_list:
    print(f"running for n={n_bodies}")
    times[j] = []
    
    for i in range(repeats):
        start = time.perf_counter()
        
        state0 = iniconf.N_horizontal(n_bodies)
        Y = N_body_pendulum(n_bodies, link, tspan, state0)
        
        end = time.perf_counter()
        times[j].append(end - start)
    j += 1
    
    
data_for_export = []
for n_bodies, run_times in times.items():
    for duration in run_times:
        data_for_export.append({"n_bodies": n_bodies, "duration": duration})

# 2. Create DataFrame and Save
df = pd.DataFrame(data_for_export)
df.to_csv("pendulum_benchmark.csv", index=False)

print("Data saved to pendulum_benchmark.csv")

print("done")
