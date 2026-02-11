import numpy as np
def rotfromquat(quat):
    #Convert a quaternion to a rotation matrix.

    #Args: quat: Unit quaternion as a np array of shape (4,)

    #Returns: np array: A 3x3 rotation matrix a np array of shape (3,3)

    assert quat.shape == (4,), "Input quaternion must be of shape (4,)"
    
    I = np.eye(3)
    q0 = quat[3]
    q = quat[:3]
    q_tilde = skewfromvec(q)
    R = I + 2*(q0*I + q_tilde)@q_tilde
    return R

def skewfromvec(vec):
    #Convert a 3D vector to a skew-symmetric matrix.

    #Args: vec: A 3D vector as a np array of shape (3,)

    #Returns: np array: A 3x3 skew-symmetric matrix as a np array of shape (3,3)
    
    assert vec.shape == (3,), "Input vector must be of shape (3,)"

    S = np.array([[0, -vec[2], vec[1]],
                      [vec[2], 0, -vec[0]],
                      [-vec[1], vec[0], 0]])
    return S

def RBT(vec):
    #Rigid body transformation matrix - no rotations

    # Args: 3D vector as a np array of shape (3,)

    # Returns: 6x6 rigid body transformation matrix as a np array of shape (6,6)  
    
    assert vec.shape == (3,), "Input vector must be of shape (3,)"

    I = np.eye(3)
    Z = np.zeros((3,3))
    l_tilde = skewfromvec(vec)

    phi = np.block([[I,l_tilde],[Z,I]])
    return phi

def spatialskewbar(X):
    #Convert a 6D spatial-vector into a 6x6 skew-symmetric matrix as defined in 1.25 in ABI book

    #Args: vec: A 6D vector as a np array of shape (6,)

    #Returns: np array: A 6x6 skew-symmetric matrix as a np array of shape (6,6)

    assert X.shape == (6,), "Input spatial vector must be of shape (6,)"

    X_bar = np.block([[skewfromvec(X[:3]),skewfromvec(X[3:])],
             [np.zeros((3,3)),skewfromvec(X[:3])]])
    return X_bar

    
def spatialskewtilde(spatialvec):
    #Convert a 6D spatial vector to a 6x6 skew-symmetric matrix.

    #Args: spatialvec: A 6D spatial vector as a np array of shape (6,)

    #Returns: np array: A 6x6 skew-symmetric matrix as a np array of shape (6,6)

    assert spatialvec.shape == (6,), "Input spatial vector must be of shape (6,)"

    w = spatialvec[:3]
    v = spatialvec[3:]

    w_tilde = skewfromvec(w)
    v_tilde = skewfromvec(v)

    S = np.block([[w_tilde, np.zeros((3,3))],
                  [v_tilde, w_tilde]])
    return S
    
def spatialrotfromquat(quat):
    #Convert a quaternion to a 6x6 spatial rotation matrix.

    #Args: quat: Unit quaternion as a np array of shape (4,)

    #Returns: np array: A 6x6 spatial rotation matrix as a np array of shape (6,6)

    assert quat.shape == (4,), "Input quaternion must be of shape (4,)"
    
    R = rotfromquat(quat)

    spatialR = np.block([[R, np.zeros((3,3))],
                         [np.zeros((3,3)), R]])
    return spatialR

def derrivmap(theta,omega,type="type of joint"):
    #arguments:
    #theta: scalar or np array of shape (N,). Generalized coordinate(s)
    #omega: angular velocity of k wrt to k+1 (this is usually just the generalized velocity depending on definition chosen)
    #type: string indicating the type of hinge, either "revolute" or "spherical)

    #returns:
    #derrivative of generalized coordiantes (theta_dot) - flattened, i.e of shape (N,)

    omega = omega.reshape(3,1)
    
    if type == "revolute":
        derriv = omega
    elif type == "spherical":
        derriv = 0.5*np.block([[-skewfromvec(omega.flatten()), omega],
                               [-omega.T, 0]]) @ theta.reshape(4,1)

    else:
        raise ValueError("Type must be either 'revolute' or 'spherical'")
    
    return derriv.flatten()

def quatfromrev(theta,axis="axis of orientation"):
    #computes the quartenion represenation of the relative rotation of two bodies for a revolute joint
    #arguments:
    #theta: scalar: Generalized coordinate for that revolute joint
    #axis: Rotation axis as a string, either "x", "y" or "z"

    #returns:
    #quat: Unit quaternion as a np array of shape (4,)

    if axis == "x":
        n = np.array([1,0,0])
    elif axis == "y":
        n = np.array([0,1,0])
    elif axis == "z":
        n = np.array([0,0,1])
    else:
        raise ValueError("Axis must be either 'x', 'y' or 'z'")
    
    q_vec = np.sin(theta/2)*n
    q_scalar = np.cos(theta/2)

    q = np.concatenate((q_vec, np.array([q_scalar])))

    return q

class SimpleLink:
    def __init__(self,m,l_hinge):

        #adding attribtues to object
        self.m = m
        self.l_com = l_hinge/2
        self.l_hinge = l_hinge

        #calculating geometry (right now width and heigh of link is just 1/10 of length)
        l = np.linalg.norm(l_hinge)
        w = l/10
        h = w
        self.J_c = np.diag([1/12*m*(h**2 + w**2), 1/12*m*(l**2 + h**2), 1/12*m*(l**2 + w**2)])

        #spatial inertia at COM
        self.M_c =  np.block([[self.J_c, np.zeros((3,3))],
                          [np.zeros((3,3)), m*np.eye(3)]])
    
        self.M = RBT(self.l_com)@self.M_c@RBT(self.l_com).T #spatial inertia at body frame (located at hinge)

        #rigidbody transform across link
        self.RBT = RBT(l_hinge)
    
    def set_hingemap(self,type="hingetype"):
        if type == "spherical":
            self.H =np.block([[np.eye(3), np.zeros((3,3))]])
        else:
            print("right now i have only specified for spherical joints")

def normalize_quaternions(q):
    #takes a vector of stacked quartenions and normalized them. fully vectorized.
    q = np.asarray(q)
    q_reshaped = q.reshape(-1, 4)
    norms = np.linalg.norm(q_reshaped, axis=1, keepdims=True)
    q_reshaped /= norms
    return q_reshaped.reshape(-1)
        
def ATBI_N_body_pendulum(state,tau_vec,n,link):
        #inputs
        #state: np.array on form [theta_dot, beta]
        #tau_vec: generalized forces as np.array
        #l_hinge: vector from O_k to O+_k-1 in k frame (this doesnt matter as they are identical in our case)
        #m: mass of length. Ensure that you dont have a very long and slender link with a small mass to avoid very stiff elements
        #type: hinge-type for all links. Right now its purely spherical that is implemented
        #n: no_bodies
        #link: Instantiate a link using the SimpleLink class and pass it
        #outputs beta_dot

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

        #unpacking interior 
        for i in range(1, n+1):

            idxq = 4*(i-1)
            idxw = 3*(i-1)

            theta[i] = theta_vec[idxq:idxq+4]
            beta[i]  = beta_vec[idxw:idxw+3]
            tau[i]   = tau_vec[3*(i-1):3*i]

         #if damping is to be implemented, then add a -b*beta[i] component in the for loop, or just do a simple tau[n] = -b*tau[n] if you only wish to damp body attatched to ground   

        #storage
        P_plus = [None]*(n+2)
        xi_plus = [None]*(n+2)
        nu = [None]*(n+2)
        A = [None]*(n+2)
        V = [None]*(n+2)
        G = [None]*(n+2)
        D = [None]*(n+2)
        beta_dot = [None]*(n+2)
        tau_bar = [None]*(n+2)
        agothic = [None]*(n+2)
        bgothic = [None]*(n+2)
        
        #gravity and storage of gravity
        g = [None]*(n+2)
        g[n+1] = np.array([0,0,0,0,0,9.81]) #in inertial frame 

        #boundary conditions on spatial operator quantities
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = P_plus[0]
        A[n+1] = np.array([0, 0, 0, 0, 0, 0])
        V[n+1] = np.zeros((6,))

        #kinematics scatter

        for k in range(n,0,-1):
            #rotation matrices
            pRc = spatialrotfromquat(theta[k]) 
            cRp = pRc.T #from parent to child -> this is the direction we are going right now

            #rotating gravity such that we have that in frame aswell
            g[k] = cRp@g[k+1]

            #hinge contribtuion
            delta_V = link.H.T @ beta[k]

            #spatial velocity
            V[k] = cRp @ link.RBT.T @ V[k+1] + delta_V

            #coriolois acc
            agothic[k] = spatialskewtilde(V[k]) @ link.H.T @ beta[k]

            #gyroscopic term
            bgothic[k] = spatialskewbar(V[k]) @ link.M @ V[k]

        #ATBI gather 
        for k in range(1,n+1): #n+1 as python does not include end index

            #rotations
            pRc = spatialrotfromquat(theta[k-1]) #using k-1 as orientation is defined as k+1_q_k and we need k_q_k-1
            cRp = pRc.T 

            P = link.RBT @ pRc @ P_plus[k-1] @ cRp@link.RBT.T + link.M
            D[k] = link.H @ P @ link.H.T
            G[k] = np.linalg.solve(D[k], link.H @ P).T #P @ link.H.T @ np.linalg.inv(D)
            tau_bar[k] = np.eye(6) - G[k] @ link.H
            P_plus[k] = tau_bar[k] @ P
            xi = link.RBT @ pRc @ xi_plus[k-1] + P @ agothic[k] + bgothic[k]
            eps = tau[k] - link.H@xi
            nu[k] = np.linalg.solve(D[k], eps) #= np.linalg.inv(D)@eps
            xi_plus[k] = xi + G[k]@eps

        #ATBI scatter
        for k in range(n,0,-1):
            #rotations
            pRc = spatialrotfromquat(theta[k])
            cRp = pRc.T 

            A_plus = cRp@ link.RBT.T @A[k+1]
            nu_bar = nu[k] - G[k].T @ g[k]  
            beta_dot[k] = nu_bar - G[k].T @ A_plus 
            A[k] = A_plus + link.H.T @ beta_dot[k] + agothic[k]

        return A, V, beta_dot,tau_bar,D,G #which is theta_ddot depending on how you look at it
    
def omega(link,tau_bar,D,n):
    #calculating gamma for base link (then we dont even need a for loop)
    gamma_n = link.H.T @ np.linalg.solve(D[n], link.H)


    #starting loop
    omega = gamma_n
    gamma = np.zeros((6,6))

    for k in range(n,0,-1):
        
        psi = link.RBT @ tau_bar[k]
        omega = omega @ psi

        gamma = psi.T@gamma@psi +link.H.T @ np.linalg.solve(D[k], link.H)

    omega_n1 = omega
    omega_nn = gamma_n
    omega_11 = gamma

    return omega_11, omega_nn, omega_n1

def beta_dot_delta(tau_bar,link,n,D,f_c,G):

    #f_c comes with RBT already applied where nessecary

    xi_delta = [None]*(n+2)
    beta_dot_delta = [None] * (n+2)
    nu = [None]*(n+2)
    lambda_list = [None]*(n+2) #NOT TO BE CONFUSED WITH LAGRANGE MULTIPLIERS; THIS IS JUST THE NOTATION FROM THE BOOK


    #boundary cond on xi_delta and lambda_list
    xi_delta[0] = np.zeros(6,1)
    lambda_list[n+1] = np.zeros(6,1)

    for k in range (0,n+1):
        psi = link.RBT @ tau_bar[k-1]
        xi_delta[k] = psi@xi_delta[k-1] - f_c[k]

        nu[k] = -np.linalg.solve(D[k],link.H@xi_delta[k])

    for k in range(n,0,-1):
        psi = link.RBT @ tau_bar[k]
        kappa = link.RBT @ G[k]

        lambda_list[k] = psi.T@lambda_list[k+1] + link.H.T@nu[k]


        beta_dot_delta[k] = nu[k] - kappa.T@lambda_list[k+1]

    return beta_dot_delta


    


        