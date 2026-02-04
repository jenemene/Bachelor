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

    

    

