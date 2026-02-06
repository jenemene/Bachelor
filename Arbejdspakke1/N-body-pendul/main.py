
import matplotlib.pyplot as plt 
import numpy as np
import soa as SOA
from scipy.integrate import solve_ivp
import plotting as SOAplt

def N_body_pendulum(n):

    #mangler inputs fra tavlebillede, dvs ting som m, l_link,l_com, J osv osv. De kan implementeres når egen solver er oppe at køre.

    state0 = initial_config(n)

    def odefun(t,state,n):
        #solve_ivp passes state as np.array. It is unpacked, and then passed to ATBI as a list of form state = [theta,beta]

        #space allocation for lists
        theta = [None] * n
        beta = [None] * n
        theta_dot = [None] * n
        beta_dot = [None] * n

        #unpack state and calculation of theta_dot
        for i in range(n):
            idxq = 4*i
            idxw = 4*n + 3*i
            theta[i] = state[idxq:idxq+4]
            theta[i] = theta[i] / np.linalg.norm(theta[i]) # Force unit length
            beta[i] = state[idxw:idxw+3]

            theta_dot[i] = SOA.derrivmap(theta[i],beta[i],"spherical")
        
        #calculation of beta_dot (ATBI)
        tau = np.zeros_like(beta) #no external torques
        _, beta_dot = ATBI(theta, beta, tau)

        state_dot = np.concatenate(theta_dot + beta_dot)

        return state_dot

    def ATBI(theta, beta, tau):
        #vectors
        l_hinge = np.array([0,0,5]) #vector from hinge to hinge in link frame. This is vector l(O_k,O+_k-1) in k frame.
        l_com = np.array([0,0,2.5]) #vector from hinge to com in link frame. This is vector l(O_k,C_k) in k frame.

        #geometry of link
        l = np.linalg.norm(l_hinge) #length of link
        w = l/10 #m - width of link
        h = l/10 #m - height of link

        #mass and inertia matrix
        m = 200 #kg - very heavy link
        J_c = np.diag([1/12*m*(h**2 + w**2), 1/12*m*(l**2 + h**2), 1/12*m*(l**2 + w**2)]) #inertia matrix at com in link frame

        #spatial inertia matrix at COM
        M_c = np.block([[J_c, np.zeros((3,3))],
                          [np.zeros((3,3)), m*np.eye(3)]])
        
        #spatial inertia at hinge
        M = SOA.RBT(l_com)@M_c@SOA.RBT(l_com).T #hvis der er en fejl så kig her først

        #rigid body transform across links (from parent inboard to outboard)
        RBT = SOA.RBT(l_hinge)
        RBT_com = SOA.RBT(-l_com)

        #Hingemap
        H = np.block([[np.eye(3), np.zeros((3,3))]])

        # Spatial force from grativy
        f_gravity = np.array([0,0,0,0,0,-m*9.81])

        #Gravity in inertial
        g_inertial = np.array([0,0,0,0,0,-9.81])
        
        #Kinematics scatter!
        # Preparing cells for values to store:
        V = [None] * n
        agothic = [None] * n
        bgothic = [None] * n
        g = [None] * n #accelerations in local body frame
        nRI = [None] * n

        for k in reversed(range(n)):
            #rotation matrices
            pRc = SOA.spatialrotfromquat(theta[k]) #rotation from child to parent
            cRp = pRc.T #rotation from parent to child

            nRI[k] = nRI[k+1] @ cRp if k < n-1 else cRp #rotation from inertial to parent, if k = n-1 then it is identity since the parent is inertial frame

            #Spatial velocities
            delta_V = H.T @ beta[k] #hinge contribution
           
            if k == n-1: #special case, last hinge. No spatial velocity of inertial frame (it is inertial :))
                V[k] = delta_V
            else:
                 V[k] = cRp @ RBT.T @ V[k+1] + delta_V ### Added RBT.T
                
            # Coriolis term (const. joint map and pure rotation):
            if k == n - 1:
                agothic[k] = SOA.spatialskewtilde(V[k]) @ delta_V - SOA.spatialskewbar(delta_V) @ delta_V
            else:
                agothic[k] = SOA.spatialskewtilde(V[k]) @ delta_V

            # Gyroscopic term:
            bgothic[k] = SOA.spatialskewbar(V[k]) @ M @ V[k]

        #First ATBI sweep - Gather ATBI quanteties
        # Preparing cells for values to store:
        Pplus = [None] * n
        G = [None] * n
        nu = [None] * n
        nu_bar = [None] * n
        varepsplus = [None] * n
        alpha = [None] * n
        gamma = [None] * n


        for k in range(n):

            if k == 0:
                P = M
                D = H @ P @ H.T
                G[k] = P @ H.T @ np.linalg.inv(D)
                taubar = np.eye(6) - G[k] @ H
                Pplus[k] = taubar @ P
                vareps = P @ agothic[k] + bgothic[k] - RBT_com @ nRI[k] @ f_gravity 
                eps = tau[k] - H @ vareps 
                nu[k] = np.linalg.inv(D) @ eps
                varepsplus[k] = vareps + G[k] @ eps
            else:
                # Rotation from parent to child:
                pRc = SOA.spatialrotfromquat(theta[k-1])
                # Rotation from child to parent:
                cRp = pRc.T

                P = RBT @ pRc @ Pplus[k-1] @ cRp @ RBT.T + M
                D = H @ P @ H.T
                G[k] = P @ H.T @ np.linalg.inv(D)
                taubar = np.eye(6) - G[k] @ H
                Pplus[k] = taubar @ P
                vareps = RBT @ pRc @ varepsplus[k-1] + P @ agothic[k] + bgothic[k] - RBT_com @ nRI[k] @ f_gravity
                eps = tau[k] - H @ vareps 
                nu[k] = np.linalg.inv(D) @ eps
                varepsplus[k] = vareps + G[k] @ eps

        #second ATBI sweep - 
        # ATBI scatter (using if statement to enforce alpha(n+1)=0):
        for k in reversed(range(n)):
            if k == n - 1:
                alphaplus = np.zeros(6)
                nu_bar[k] = nu[k]
                gamma[k] = nu_bar[k] - G[k].T @ alphaplus
                alpha[k] = alphaplus + H.T @ gamma[k] + agothic[k]
            else:
                # Rotation from child to parent:
                pRc = SOA.spatialrotfromquat(theta[k])
                # Rotation from parent to child:
                cRp = pRc.T
                
                #loop itself
                alphaplus = cRp @ RBT.T @ alpha[k+1]
                nu_bar[k] = nu[k]
                gamma[k] = nu_bar[k] - G[k].T @ alphaplus
                alpha[k] = alphaplus + H.T @ gamma[k] + agothic[k]
        return alpha, gamma #gamma = beta_dot

    # Solve the ODE using scipy's solve_ivp
    tspan = np.arange(0, 60,0.1)
    result = solve_ivp(
    odefun, 
    t_span=(0, tspan[-1]), 
    y0=state0, 
    method='Radau', 
    t_eval = tspan,
    args=(n,)
    )
    # Extract time and state vectors
    return result

def initial_config(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(-np.pi/2, "y")
    q_rest = np.array([0,0,0,1])
    q_rest_tiled = np.tile(q_rest, n-1)
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    zeros = np.zeros(3 * n)
    
    # Concatenate into one long state vector
    state0 = np.concatenate([q_rest_tiled, qn, zeros])

    return state0

n_bodies = 1

result = N_body_pendulum(n_bodies)
print(result)

SOAplt.N_body_pendulum_gen_plot(result.t,result.y,n_bodies)

