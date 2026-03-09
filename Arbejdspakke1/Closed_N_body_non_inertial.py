import numpy as np
from utils import soa as SOA
from utils import plotting as SOAplt
import time
from initial_configs import initial_configs_closed as iniconf

def N_body_pendulum_closed(n, link, tspan, state0, BG_params):
    start = time.perf_counter()
    print("--- Simulation started ---")
    print(f"Progress (until {tspan[-1]:.0f}):")

    def ODEfun(t, state, n, link, BG_params):
        #solve_ivp passes state as np.array. It is unpacked, and then passed to ATBI as a a list of form state = [theta,beta].

        #unpacking state
        theta = state[:4*n]
        beta = state[4*n:]

        #normalizing quartenions
        theta = SOA.normalize_quaternions(theta) 
        
        #calculating theta_dot based on the derrivmap function
        theta_dot = np.zeros(len(theta))
        for i in range(n):
            idxq = 4*i #these indexes assume that we ONLY have spherical joints
            idxw = 3*i
            theta_dot[idxq:idxq+4] = SOA.derrivmap(theta[idxq:idxq+4],beta[idxw:idxw+3],"spherical")
            
        #Calculationg of generalized accelerations without any constraints (beta_dot_free) - this requires ATBI. 
        tau_vec = np.zeros_like(beta) #no external torques

        A_f, V_f, beta_dot_f_list, tau_bar, D, G = SOA.ATBI(state, tau_vec, n, link)

        beta_dot_f = np.concatenate([b.flatten() for b in beta_dot_f_list[1:n+1]])

        #Calculation of A_nd (V_nd is not needed as Q is constant) 

        IR1 = SOA.get_rotation_tip_to_body_I(theta, n) #rotations to to ensure we are consistent with frames
        IRn = SOA.spatialrotfromquat(theta[4*(n-1):4*(n-1)+4])
        #A_nd = np.concatenate([IRn @ A_f[n],IR1 @ link.RBT.T @ A_f[1]]) # Hvis denne bruges, så tjek her om den er i rigtig rækkefølge ift. Q og udledning.

        
        #Setting up Q
        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d,-d])


        #need to calculate LAMDA (the matrix thing). For that we need elements of OMEGA
        omega_nn, omega_n1, omega_1n, omega_11= SOA.omega(theta,link,tau_bar,D,n)

        # Calculating block entries JEG ER MEGET I TVIL HER, MEN VED AT LAMBDA_11 er god nok :)
        Λ_11 = link.RBT.T @ omega_11 @ link.RBT
        Λ_nn = omega_nn
        Λ_n1 = omega_n1 @ link.RBT
        Λ_1n = link.RBT.T @ omega_1n

        # Rotate everything to the Inertial frame (has to be done on both sides)
        Λ_11 = IR1 @ Λ_11 @ IR1.T
        Λ_nn = IRn @ Λ_nn @ IRn.T
        Λ_n1 = IR1 @ Λ_n1 @ IR1.T 
        Λ_1n = IR1 @ Λ_1n @ IR1.T 

        # Build the 12x12 block matrix
        Λ_block = np.block([
            [Λ_nn, Λ_n1],
            [Λ_1n, Λ_11]
        ])

        positions = SOA.compute_pos_in_inertial_frame(state, link.l_hinge, n)

        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  l_IOn - (l_IO1 + IR1[:3, :3]@link.l_hinge)
        Φ_dot = IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link.l_hinge)
        Φ_ddot =  IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link.l_hinge + IωIO@IωIO@IR1[:3,:3]@link.l_hinge)

        #---------------------------------PRINTING HERE---------------------------------------#
        print(f"t={t:.2f}  |Φ| = {np.linalg.norm(Φ):.8f}")

        # Baumgarte stabilization
        α, β = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β) # Parametrene er vi slet ikke sikker på) AYO HVORFOR FUCK HEDDER DEN F

        #solving for lagrange multipliers
        λ = -np.linalg.solve((Q@Λ_block@Q.T),f) # Dimension: 3x1

        #calculating f_c
        f_c_closed_loop_const =  Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        f_c[n] = IRn.T @ f_c_closed_loop_const[:6]
        f_c[1] = link.RBT @ IR1.T @ f_c_closed_loop_const[6:] 

        #calculating beta_dot_delta
        beta_dot_delta_list = SOA.beta_dot_delta(theta,tau_bar,link,n,D,f_c,G) #returns a list

        beta_dot_delta = np.concatenate([b.flatten() for b in beta_dot_delta_list[1:n+1]])

        beta_dot = beta_dot_delta + beta_dot_f

        state_dot = np.concatenate([theta_dot, beta_dot.flatten()])

        return state_dot, V_f
    
    result, V_values = SOA.RK4_int_with_V_BG(ODEfun, state0, tspan, n, link, BG_params)

    end = time.perf_counter()
    print(f"Integration time: {end - start:.2f} seconds")
    print("--- Simulation finished ---")

    return result, V_values, link

### LINK SETUP ###
m = 20
l_hinge = np.array([0,0,0.2])
link = SOA.SimpleLink(m, l_hinge)
link.set_hingemap("spherical")

### SIMULATION SETTINGS ###
n_bodies = 3
simulation_length = 5
dt = 0.001
state0 = iniconf.N3_triangle(n_bodies)

### PLOT INITIAL STATE ###
SOAplt.plot_initial_state(state0, link, config="closed")

### BAUMGARTE PARAMETERS ###
α = 2000
β = 2500
BG_params = np.array([α, β])

### RUN SIMULATION ###
tspan = np.arange(0, simulation_length+dt, dt)
states, V_values, link = N_body_pendulum_closed(n_bodies, link, tspan, state0, BG_params)

### ANIMATION ###
SOAplt.animation_plot(states, tspan, link, config="closed", step=30)

### ENERGY CHECK ###
SOAplt.check_energies(states, V_values, tspan, link, n_bodies, "closed_3", TE_only=False)
SOAplt.check_energies(states, V_values, tspan, link, n_bodies, "closed_3", TE_only=True)


