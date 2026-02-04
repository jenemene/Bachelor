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