import numpy as np

def rotfromquat(quat):
    #Convert a quaternion to a rotation matrix.

    #Args: quat: Unit quaternion as a np array of shape (4,1)

    #Returns: np array: A 3x3 rotation matrix a np array of shape (3,3)
    
        I = np.eye(3)
        q0 = quat[3]
        q = quat[:3].reshape(3,1)
        q_tilde = skewfromvec(q)
        R = I + 2*(q0*I + q_tilde)@q_tilde
        return R

def skewfromvec(vec):
    #Convert a 3D vector to a skew-symmetric matrix.

    #Args: vec: A 3D vector as a np array of shape (3,1)

    #Returns: np array: A 3x3 skew-symmetric matrix as a np array of shape (3,3)
    
    vec = vec.flatten()

    S = np.array([[0, -vec[2], vec[1]],
                      [vec[2], 0, -vec[0]],
                      [-vec[1], vec[0], 0]])
    return S

def rigidbodytransform(vec):
    #Create a rigid body transformation matrix from a 3D translation vector.

    #Args: vec: A 3D translation vector as a np array of shape (3,1)

    #Returns: np array: A 6x6 rigid body transformation matrix as a np array of shape (6,6)

    l = vec.flatten()

    l_tilde = skewfromvec(vec)

    I = np.eye(3)

    phi = np.block([[I, l_tilde],
                    [np.zeros((3,3)), I]])
    return phi

def skewfromspatialvec(spatialvec):
    #Convert a 6D spatial vector to a 6x6 skew-symmetric matrix.

    #Args: spatialvec: A 6D spatial vector as a np array of shape (6,1)

    #Returns: np array: A 6x6 skew-symmetric matrix as a np array of shape (6,6)

    spatialvec = spatialvec.flatten()

    w = spatialvec[:3]
    v = spatialvec[3:]

    w_tilde = skewfromvec(w.reshape(3,1))
    v_tilde = skewfromvec(v.reshape(3,1))

    S = np.block([[w_tilde, np.zeros((3,3))],
                  [v_tilde, w_tilde]])
    return S

def skewbarfromspatialvec(spatialvec):
    #Convert a 6D spatial vector to a 6x6 skew-symmetric matrix in the bar representation.

    #Args: spatialvec: A 6D spatial vector as a np array of shape (6,1)

    #Returns: np array: A 6x6 skew-symmetric matrix as a np array of shape (6,6)

    spatialvec = spatialvec.flatten()

    w = spatialvec[:3]
    v = spatialvec[3:]

    w_tilde = skewfromvec(w.reshape(3,1))
    v_tilde = skewfromvec(v.reshape(3,1))

    S_bar = np.block([[w_tilde, v_tilde],
                      [np.zeros((3,3)), w_tilde]])
    return S_bar

def spatialrotfromquat(quat):
    #Convert a quaternion to a 6x6 spatial rotation matrix.

    #Args: quat: Unit quaternion as a np array of shape (4,1)

    #Returns: np array: A 6x6 spatial rotation matrix as a np array of shape (6,6)
    
    R = rotfromquat(quat)

    spatialR = np.block([[R, np.zeros((3,3))],
                         [np.zeros((3,3)), R]])
    return spatialR