import matplotlib.pyplot as plt 
import numpy as np
import soa as SOA
from scipy.integrate import solve_ivp
import plotting as SOAplt

def N_body_pendulum_2(t,state,n):
    
    state0 = initial_config(n)

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
        idxw = 4*n + 3*i
        theta_dot[idxq:idxq+4] = SOA.derrivmap(theta[idxq:idxq+4],beta[idxw:idxw+3],"spherical")
        
    #Calculationg of generalized accelerations (beta_dot) - this requires ATBI. 
    tau_vec = np.zeros_like(beta) #no external torques

    A,V,beta_dot = ATBIalg(state,tau_vec,n)

    state_dot = np.concatenate(theta_dot + beta_dot)

    return state_dot

def ATBIalg(state,tau_vec,n):
    #setting up link
    m = 20
    l_com = np.array([0,0,2.5])
    l_hinge = np.array([0,0,5])
    link = SOA.SimpleLink(m,l_com,l_hinge)
    link.set_hingemap("spherical")

    #rigidbodytransform
    RBT = SOA.RBT(l_hinge)
    RBT_com = SOA.RBT(l_hinge)
    #unpacking state

    theta_vec = state[:4*n]
    beta_vec  = state[4*n:]

    theta = [None]*(n+2)
    beta  = [None]*(n+2)
    tau   = [None]*(n+2)

    # boundary conditions - det kan diskuteres om man behøver i begge ender for dem alle, det gør man vidst nok ikke
    theta[0]   = np.zeros(4)
    theta[n+1] = np.zeros(4)

    beta[0]    = np.zeros(3)
    beta[n+1]  = np.zeros(3)

    tau[0]     = np.zeros(3)
    tau[n+1]   = np.zeros(3)

    #unpacking interior (IDK OM VI SKAL PASSE DOM EN FUCKING LISTE JEG FØLGER MIT RETARD MATLAB)
    for i in range(1, n+1):

        idxq = 4*(i-1)
        idxw = 3*(i-1)

        theta[i] = theta_vec[idxq:idxq+4]
        beta[i]  = beta_vec[idxw:idxw+3]
        tau[i]   = tau_vec[3*(i-1):3*i]

        

    #storage
    P_plus = [None]*(n+2)
    xi_plus = P_plus
    nu = P_plus
    A = P_plus
    V = P_plus
    G = P_plus
    beta_dot = P_plus
    tau_bar = P_plus
    agothic = [None] * (n+2)
    bgothic = [None] * (n+2)
    kRI = [None] * (n+2)

    #boundary conditions on spatial operator quantities
    P_plus[0] = np.zeros((6,6))
    xi_plus[0] = np.zeros((6,1))
    tau_bar[0] = P_plus[0]
    A[n+1] = np.zeros((6,1))
    V[n+1] = np.zeros((6,1))
    kRI[n+1] = np.eye(3)

    for k in range(n,0,-1):
        #rotation matrices
        pRc = SOA.spatialrotfromquat(theta[k])
        cRp = pRc.T #from parent to child -> this is the direction we are going right now

        #commultative rotation
        kRkp1 = SOA.rotfromquat(theta[k])

        kRI[k] = kRkp1 @ nRI[k+1] #jeg bliver i tvivl nu om jeg overhovedet behandler body n som inertial eller om jeg nu rent faktsik behandler body n+1 som inertial

        #hinge contribtuion
        delta_V = link.H.T @ beta[k]

        #spatial velocity
        V[k] = cRp @ RBT.T @ V[k+1] + delta_V

        #coriolois acc
        agothic[k] = SOA.spatialskewtilde(V[k]) @ link.H.T @ beta[k]

    #ATBI gather KOM I GANG






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