import numpy as np
import soa_jens as SOA
#from scipy.integrate import solve_ivp
import jens_plotting as SOAplt
import time
import matplotlib.pyplot as plt

def N_body_pendulum_closed(n):
    def ODEfun(t,state,n,link):
        #solve_ivp passes state as np.array. It is unpacked, and then passed to ATBI as a a list of form state = [theta,beta].

        #unpacking state
        theta = state[:4*n]
        beta = state[4*n:]

        #normalizing quartenions
        #theta = SOA.normalize_quaternions(theta) 
        
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
        #man kan lave en A_stacked her hvis man vil A_st = A_n, A_1. Det gider jeg sku ik

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
        
        
        #print(f"t={t:.2f}  |Φ| = {np.linalg.norm(Φ):.6f}")

        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, 20, 20) # Parametrene er vi slet ikke sikker på) AYO HVORFOR FUCK HEDDER DEN F

        #solving for lagrange multipliers
        λ = -np.linalg.solve((Q@Λ_block@Q.T),f) # Dimension: 3x1


        #calculating f_c (skal ændred noget her)
        f_c_closed_loop_const =  Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]


        f_c[n] = IRn.T @ f_c_closed_loop_const[:6]
        f_c[1] = link.RBT @ IR1.T @ f_c_closed_loop_const[6:] # SKAL VÆRE SÅDAN HER!!!

        print(t)
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
        return state_dot
        

    
    #setting up link
    m = 2 #mass in kg
    l_hinge = np.array([0,0,0.2])
    link = SOA.SimpleLink(m,l_hinge)
    link.set_hingemap("spherical")

    #initial config.
    state0 = N4_initial_config(n)
    
    tspan = np.arange(0, 10, 0.001)
    result = SOA.RK4_int(ODEfun, state0, tspan, n,link)

    return result,tspan,link
    

#ONLY for 4 links right now due to initial config.
def N4_initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(np.pi/2, "y")
    q_all = np.tile(qn, n)
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    ωn = np.array([0,np.pi/2,0])
    ω1 = np.zeros(3)
    ω1_tiled = np.tile(ω1, n-1)
    ω_all = np.concatenate([ω1_tiled, ωn])*0 # <------------------- Jeg har lige sat den til 0 :)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N4_stardown_initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(np.pi/4, "y")
    q_other = SOA.quatfromrev(-np.pi/2, "y")
    q_other_all = np.tile(q_other, n-1)
    q_all = np.concatenate([q_other_all, qn])
    
    # Create zero vector for initial velocities  
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N4_starup_initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(3*np.pi/4, "y")
    q_other = SOA.quatfromrev(np.pi/2, "y")
    q_other_all = np.tile(q_other, n-1)
    q_all = np.concatenate([q_other_all, qn])
    
    # Create zero vector for initial velocities  
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

# ONLY for 2 links right now due to initial config.
def N2_initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(np.pi/2, "y")
    q1 = SOA.quatfromrev(np.pi, "y")
    q_all = np.concatenate([q1, qn])
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    ωn = np.array([0,np.pi,0])
    ω1 = np.zeros(3)
    ω_all = np.concatenate([ω1, ωn])*0 # <------------------- Jeg har lige sat den til 0 :)
    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

n_bodies = 4

start = time.perf_counter()

result,tspan,link = N_body_pendulum_closed(n_bodies)

end = time.perf_counter()


# Extract the state matrix (Shape: [states, time_steps])
y_out = result
    
# Clean up any microscopic quaternion drift in the final output
for i in range(len(tspan)):
    # Ensure we are using your safe, non-mutating normalize_quaternions function
    y_out[:4*n_bodies, i] = SOA.normalize_quaternions(y_out[:4*n_bodies, i])


#til animation
step = 5

t_anim = tspan[::step]
y_anim = y_out[:, ::step]

SOAplt.animation_plot(result, tspan, link, config="closed")

print("========================================================================================")
print(f"Simulation time: {end - start:.4f} seconds")
print("========================================================================================")

