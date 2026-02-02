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