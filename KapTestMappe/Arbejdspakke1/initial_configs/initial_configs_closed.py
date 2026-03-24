import numpy as np
from utils import soa as SOA

def N2_crank(n):
    # Calculate initial config for n bodies
    q2 = SOA.quatfromrev(5*np.pi/6, "y")
    q1 = SOA.quatfromrev(8*np.pi/6, "y")
    q_all = np.concatenate([q1, q2])

    # Create the zero vectors for the other initial velocities states (n, 3)
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N3_bar(n):
    # Calculate initial config for n bodies
    q3 = SOA.quatfromrev(5*np.pi/4, "y")
    q2 = SOA.quatfromrev(5*np.pi/4, "y")
    q1 = SOA.quatfromrev(7*np.pi/4, "y")
    q_all = np.concatenate([q1, q2, q3])

    # Create the zero vectors for the other initial velocities states (n, 3)
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N4_square(n):
    # Calculate initial config for n bodies
    # q0: All aligned and tilted to some side
    qn = SOA.quatfromrev(np.pi/2, "y")
    q_all = np.tile(qn, n)
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    ωn = np.array([0,0,0])
    ω1 = np.zeros(3)
    ω1_tiled = np.tile(ω1, n-1)
    ω_all = np.concatenate([ω1_tiled, ωn])

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N3_triangle(n):
    # Calculate initial config for n bodies
    q3 = SOA.quatfromrev(1*np.pi, "y")
    q2 = SOA.quatfromrev(2*np.pi/3, "y")
    q1 = SOA.quatfromrev(2*np.pi/3, "y")
    q_all = np.concatenate([q1, q2, q3])

    # Create the zero vectors for the other initial velocities states (n, 3)
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N5_triangle(n):
    # Calculate initial config for n bodies
    q5 = SOA.quatfromrev(np.pi, "y")
    q4 = SOA.quatfromrev(0, "y")
    q3 = SOA.quatfromrev(np.pi-1.318, "y")
    q2 = SOA.quatfromrev(np.pi-1.318, "y")
    q1 = SOA.quatfromrev(0, "y")
    q_all = np.concatenate([q1, q2, q3, q4, q5])

    # Create the zero vectors for the other initial velocities states (n, 3)
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N9_triangle(n):
    # Calculate initial config for n bodies
    q9 = SOA.quatfromrev(np.pi, "y")
    q8 = SOA.quatfromrev(0, "y")
    q7 = SOA.quatfromrev(0, "y")
    q6 = SOA.quatfromrev(2*np.pi/3, "y")
    q5 = SOA.quatfromrev(0, "y")
    q4 = SOA.quatfromrev(0, "y")
    q3 = SOA.quatfromrev(2*np.pi/3, "y")
    q2 = SOA.quatfromrev(0, "y")
    q1 = SOA.quatfromrev(0, "y")
    
    q_all = np.concatenate([q1, q2, q3, q4, q5, q6, q7, q8, q9])
    
    # Create the zero vectors for the other initial velocities states (n, 3)
    ω = np.zeros(3)
    ω_all = np.tile(ω, n)

    # Concatenate into one long state vector
    state0 = np.concatenate([q_all, ω_all])

    return state0

def N4_stardown(n):
    # Calculate initial config for n bodies
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

def N4_starup(n):
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