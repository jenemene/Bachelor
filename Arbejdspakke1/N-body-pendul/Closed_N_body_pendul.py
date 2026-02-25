import matplotlib
from matplotlib.pylab import norm
import matplotlib.pyplot as plt 
import numpy as np
import soa as SOA
from scipy.integrate import solve_ivp
import plotting as SOAplt
import time
import initial_configs as ini_conf

def N_body_pendulum_closed(n):
    def ODEfun(t,state,n,link):
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

        A_f,V_f, beta_dot_f_list,tau_bar,D,G = SOA.ATBI_N_body_pendulum(state, tau_vec, n, link)

        beta_dot_f = np.concatenate([b.flatten() for b in beta_dot_f_list[1:n+1]])

        #Calculation of A_nd (V_nd is not needed as Q is constant) 

        IR1 = SOA.get_rotation_tip_to_body_I(theta, n) #rotations to to ensure we are consistent with frames
        IRn = SOA.spatialrotfromquat(theta[4*(n-1):4*(n-1)+4])
        #A_nd = np.concatenate([IRn @ A_f[n],IR1 @ link.RBT.T @ A_f[1]]) # Hvis denne bruges, så tjek her om den er i rigtig rækkefølge ift. Q og udledning.

        #Setting up Q
        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d])


        #need to calculate LAMDA (the matrix thing). For that we need elements of OMEGA
        omega_nn, omega_n1, omega_1n, omega_11= SOA.omega(theta,link,tau_bar,D,n)

        #calculating block entires and rotating to frame I
        Λ_11 =link.RBT.T @ omega_11 @ link.RBT

        Λ_block =  IR1 @ Λ_11 @IR1.T


        positions = SOA.compute_pos_in_inertial_frame(state, link.l_hinge, n)

        l_IO1 = positions[1]
        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
        
        Φ =  l_IO1 + IR1[:3, :3]@link.l_hinge
        Φ_dot = (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link.l_hinge)
        Φ_ddot = (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link.l_hinge + IωIO@IωIO@IR1[:3,:3]@link.l_hinge)

        #print(f"t={t:.2f}  |Φ| = {np.linalg.norm(Φ):.6f}")

        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, 50, 5) # Parametrene er vi slet ikke sikker på)

        #solving for lagrange multipliers
        λ = np.linalg.solve(Q@Λ_block@Q.T,f) # Dimension: 3x1


        #calculating f_c
        f_c_closed_loop_const =  - Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        f_c[1] = link.RBT @ IR1.T @ f_c_closed_loop_const # SKAL VÆRE SÅDAN HER!!!

        #calculating beta_dot_delta
        beta_dot_delta_list = SOA.beta_dot_delta(theta,tau_bar,link,n,D,f_c,G) #returns a list

        beta_dot_delta = np.concatenate([b.flatten() for b in beta_dot_delta_list[1:n+1]])

        beta_dot = beta_dot_delta + beta_dot_f

        state_dot = np.concatenate([theta_dot, beta_dot.flatten()])



        # ##-DEBUGGING ---------------------------------- 
        # if t < 1e-10:
        #     print("=== t=0 diagnostics ===")
        #     print(f"Φ:      {Φ}")
        #     print(f"|Φ|:    {np.linalg.norm(Φ):.10f}")
        #     print(f"Φ_dot:  {Φ_dot}")
        #     print(f"|Φ_dot|:{np.linalg.norm(Φ_dot):.10f}")
        #     print(f"Φ_ddot: {Φ_ddot}")
        #     print(f"|Φ_ddot|:{np.linalg.norm(Φ_ddot):.10f}")
        #     print(f"λ:      {λ}")
        #     print(f"f_c[1]: {f_c[1]}")
        #     print(f"constraint force in clobal coords:{f_c_closed_loop_const}")
        #     print(f"beta_dot_f:     {beta_dot_f}")
        #     print(f"beta_dot_delta: {beta_dot_delta}")
        #     print(f"sammenlagt acceleration:{beta_dot_f+beta_dot_delta}")
        
        print(t)
        return state_dot
        

    
    #setting up link
    m = 2 #mass in kg
    l_hinge = np.array([0,0,0.2])
    link = SOA.SimpleLink(m,l_hinge)
    link.set_hingemap("spherical")

    #initial config.
    state0 = ini_conf.N4_square(n)
    
    tspan = np.arange(0, 1, 0.001)
    #result = SOA.RK4_int(ODEfun, state0, tspan, n,link)

    # Extract time and state vectors
    
    result = solve_ivp(
        ODEfun,
        t_span=(0, tspan[-1]), 
        y0=state0, 
        method='Radau',
        t_eval=tspan,
        args=(n, link),
        rtol=1e-6,
        atol=1e-9
        )
    
    return result

n_bodies = 4

start = time.perf_counter()

result = N_body_pendulum_closed(n_bodies)

end = time.perf_counter()


# Extract the state matrix (Shape: [states, time_steps])
y_out = result.y
    
# Clean up any microscopic quaternion drift in the final output
for i in range(len(result.t)):
    # Ensure we are using your safe, non-mutating normalize_quaternions function
    y_out[:4*n_bodies, i] = SOA.normalize_quaternions(y_out[:4*n_bodies, i])


#til animation
step = 30 

t_anim = result.t[::step]
y_anim = y_out[:, ::step]

SOAplt.animate_n_bodies(t_anim, y_anim, np.array([0,0,0.2]),save_video=False)

print("========================================================================================")
print(f"Simulation time: {end - start:.4f} seconds")
print(f"Success: {result.success}")
print(f"Solver status: {result.message}")
print(f"Number of function evaluations: {result.nfev}")
print("========================================================================================")
