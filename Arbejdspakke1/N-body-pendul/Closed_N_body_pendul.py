import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import soa as SOA
from scipy.integrate import solve_ivp
import plotting as SOAplt
import time

def N_body_pendulum_closed(n):
    def ODEfun(t,state,n,link):
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
            
        #Calculationg of generalized accelerations without any constraints (beta_dot_free) - this requires ATBI. 
        tau_vec = np.zeros_like(beta) #no external torques

        A_f,V_f, beta_dot_f_list,tau_bar,D,G = SOA.ATBI_N_body_pendulum(state, tau_vec, n, link)

        beta_dot_f = np.concatenate([b.flatten() for b in beta_dot_f_list[1:n+1]])

        #Calculation of A_nd (V_nd is not needed as Q is constant) 

        IR1 = SOA.get_rotation_tip_to_body_I(theta, n) #rotations to to ensure we are consistent with frames
        IRn = SOA.spatialrotfromquat(theta[4*(n-1):4*(n-1)+4])
        #A_nd = np.concatenate([IRn @ A_f[n],IR1 @ link.RBT.T @ A_f[1]]) # Hvis denne bruges, så tjek her om den er i rigtig rækkefølge ift. Q og udledning.

        #Setting up Q. We are restricitig that the linear velocity has to be 0
        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d,-d])

        #need to calculate LAMDA (the matrix thing). For that we need elements of OMEGA
        omega_nn, omega_n1, omega_1n,omega_11= SOA.omega(theta,link,tau_bar,D,n)

        #calculating block entires and rotating to frame I
        Λ_n1 = IR1 @ omega_n1 @ link.RBT
        Λ_1n = IR1 @ link.RBT.T @ omega_1n
        Λ_11 = IR1 @ link.RBT.T @ omega_11 @ link.RBT
        Λ_nn = IRn @ omega_nn

        Λ_block = np.block([[Λ_nn,Λ_n1],
                                [Λ_1n,Λ_11]]) 
        
        #setting up d_ddot #her for u er der noget ala -*- giver plus agtigt. #Jeg er overbevist om at Q@A_nd ikke skal være der.
        #u_dot = IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link.l_hinge + SOA.skewfromvec(IR1[:3, :3]@V_f[1][:3])@SOA.skewfromvec(IR1[:3, :3]@V_f[1][:3])@IR1[:3, :3]@link.l_hinge 

        positions = SOA.compute_positions(state, link.l_hinge, n)

        l_IO1 = positions[-1]
        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
        
        Φ = - IR1[:3, :3]@(l_IO1 + link.l_hinge)

        Φ_dot = - ( IωIO@IR1[:3, :3]@(l_IO1+link.l_hinge) + IR1[:3,:3]@V_f[1][3:] )

        Φ_ddot = -( SOA.skewfromvec(IR1[:3, :3]@A_f[1][3:])@IR1[:3, :3]@(l_IO1+link.l_hinge) + IωIO@IωIO@IR1[:3, :3]@(l_IO1+link.l_hinge) + 2*IωIO@IR1[:3,:3]@V_f[1][3:] + IR1[:3,:3]@A_f[1][3:] )

        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, 20, 2) # Parametrene er vi slet ikke sikker på)

        #d_ddot = 0*Q@A_nd - (u_dot)

        #solving for lagrange multipliers
        λ = np.linalg.solve(Q@Λ_block@Q.T,f) # Dimension: 3x1

        #calculating f_c
        f_c_closed_loop_const = - Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        f_c[n] = IRn.T @ f_c_closed_loop_const[:6]
        f_c[1] = link.RBT @ IR1.T @ f_c_closed_loop_const[6:]

        #calculating beta_dot_delta
        beta_dot_delta_list = SOA.beta_dot_delta(theta,tau_bar,link,n,D,f_c,G) #returns a list
        
        beta_dot_delta = np.concatenate([b.flatten() for b in beta_dot_delta_list[1:n+1]])

        beta_dot = beta_dot_delta + beta_dot_f

        state_dot = np.concatenate([theta_dot, beta_dot.flatten()])

        return state_dot
    
    #setting up link
    m = 20 #mass in kg
    l_hinge = np.array([0,0,0.5])
    link = SOA.SimpleLink(m,l_hinge)
    link.set_hingemap("spherical")

    #initial config.
    state0 = custom_initial_config(n)
    
    tspan = np.arange(0, 10, 0.001)
    
    result = solve_ivp(
        ODEfun, 
        t_span=(0, tspan[-1]), 
        y0=state0, 
        method='DOP853',
        t_eval = tspan,
        args=(n,link)
        )
            # Extract time and state vectors
    return result

def custom_initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(np.pi/2, "y")
    q_tiled = np.tile(qn, n)
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    zeros = np.zeros(3 * n)
    
    # Concatenate into one long state vector
    state0 = np.concatenate([q_tiled, zeros])

    return state0

def initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(3*np.pi/4, "y")
    q_rest = np.array([0,0,0,1])
    q_rest_tiled = np.tile(q_rest, n-1)
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    zeros = np.zeros(3 * n)
    
    # Concatenate into one long state vector
    state0 = np.concatenate([q_rest_tiled, qn, zeros])

    return state0


#ONLY for 4 links right now due to initial config.
n_bodies = 4

start = time.perf_counter()

result = N_body_pendulum_closed(n_bodies)

end = time.perf_counter()

print(f"Simulation time: {end - start:.4f} seconds")

print(result)

SOAplt.animate_n_bodies(result.t,result.y, np.array([0,0,0.2]))