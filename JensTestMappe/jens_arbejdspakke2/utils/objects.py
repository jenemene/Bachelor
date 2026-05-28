from matplotlib.animation import FuncAnimation
import numpy as np
from utils import soa as SOA
import matplotlib.pyplot as plt
import time
import csv
from scipy.integrate import solve_ivp

class Joint:
    def __init__(self):
        self.nq = None #generalized coordinates
        self.nw = None #generalized velocities
        self.H = None #hingemap
        self.q_init = None # Needed to build the starting state
        self.w_init = None

    def get_derrivative(self,theta,beta):
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def get_spatial_rotation(self,theta):
        raise NotImplementedError("This method should be implemented by subclasses.")

class SphericalJoint(Joint):
    def __init__(self):
        super().__init__()
        self.nq = 4 #quaternion
        self.nw = 3 #angular velocity as generalized velocity
        self.H = np.block([[np.eye(3), np.zeros((3,3))]])
        self.q_init = np.array([0.0, 0.0, 0.0, 1.0])
        self.w_init = np.zeros(3)

    def get_derrivative(self,theta,beta):
        return SOA.derrivmap(theta,beta,"spherical")
     
    def get_spatial_rotation(self,theta):
        return SOA.spatialrotfromquat(theta)
    
    def get_translation(self,theta):
        return np.zeros(3,)

class RevoluteJoint(Joint):
    def __init__(self,axis):
        super().__init__()
        self.nq = 1 #angle
        self.nw = 1 #angular velocity as generalized velocity
        self.axis = axis 
        self.q_init = np.array([0.0])
        self.w_init = np.array([0.0])
        
        if axis == "x": self.H = np.array([[1,0,0,0,0,0]])
        elif axis == "y": self.H = np.array([[0,1,0,0,0,0]])
        elif axis == "z": self.H = np.array([[0,0,1,0,0,0]])

    def get_derrivative(self,theta,beta):
        return beta
        
    def get_quartenion(self,theta):
        quat = SOA.quatfromrev(theta[0],self.axis)
        return quat
        
    def get_spatial_rotation(self,theta):
        quat = self.get_quartenion(theta)
        return SOA.spatialrotfromquat(quat)
    
    def get_translation(self,theta):
        return np.zeros(3,)

class FreeJoint(Joint):
    def __init__(self):
        super().__init__()
        self.nq = 7 #quaternion + position
        self.nw = 6 #angular velocity + linear velocity as generalized velocity
        self.H = np.eye(6) 
        self.q_init = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]) #initially at origin with no rotation and translation
        self.w_init = np.zeros(6)

    def get_derrivative(self,theta,beta):
        theta_rot_dot = SOA.derrivmap(theta[:4],beta[:3],"spherical")
        rot = SOA.rotfromquat(theta[:4])
        theta_trans_dot = rot@beta[3:] 
        return np.concatenate([theta_rot_dot, theta_trans_dot])
    
    def get_spatial_rotation(self,theta):
        quat = theta[:4] #first 4 elements are quaternion
        return SOA.spatialrotfromquat(quat)
    
    def get_translation(self,theta):
        return theta[4:] #last 3 elements are translation

class Link:
    def __init__(self,mass,l_hinge,joint):
        self.m = mass
        self.l_com = l_hinge/2
        self.l_hinge = l_hinge
        self.joint = joint

        l = np.linalg.norm(l_hinge)
        w = l/50 # Width and height are 1/50th of the length. This is an arbitrary choice to give the link some thickness without dominating the inertia.
        h = w
        self.J_c = np.diag([
        1/12 * self.m * (h**2 + l**2), 
        1/12 * self.m * (w**2 + l**2), 
        1/12 * self.m * (w**2 + h**2)
        ]) #nakket fra wikipedia.

        self.M_c =  np.block([[self.J_c, np.zeros((3,3))],
                          [np.zeros((3,3)), self.m*np.eye(3)]])
    
        self.M = SOA.RBT(self.l_com)@self.M_c@SOA.RBT(self.l_com).T 

        self.RBT = SOA.RBT(l_hinge)

class MultiBodySystem:
    def __init__(self):
        self.links = []
        self.total_nq = 0
        self.total_nw = 0
        self.result = None
        self.tspan = None
        self.constraint_violation = []
        self._record_metrics = False
        self.l_from_origin = np.array([0,0,0])


    def add_link(self,link):
        self.links.insert(0, link)
        self.total_nq += link.joint.nq
        self.total_nw += link.joint.nw
    
    def get_initial_state(self):
        q0_list = [link.joint.q_init for link in self.links]
        w0_list = [link.joint.w_init for link in self.links]
        return np.concatenate(q0_list + w0_list)

    def unpack_state(self,state):
        #for unpacking state vector into a list of theta and beta
        theta_list = []
        beta_list = []

        idx_theta = 0
        idx_beta = self.total_nq
        
        for link in self.links:
            theta = state[idx_theta: idx_theta+link.joint.nq]
            
            # # Normalization safety check for quaternions
            # if link.joint.nq == 4:
            #     theta = SOA.normalize_quaternions(theta)
                
            theta_list.append(theta)
            idx_theta += link.joint.nq 

            beta = state[idx_beta:idx_beta + link.joint.nw]
            beta_list.append(beta)
            idx_beta += link.joint.nw
            
        return theta_list, beta_list

    def get_state_dot(self,t,state,V_base,A_base):
        theta_list, beta_list = self.unpack_state(state)

        theta_dot_list = []
        
        damping = 0.0 #damping if one wants this
        tau_list = [-damping * beta for beta in beta_list]

        for i in range(len(self.links)): #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION! APPEND IS NOT EFFICIENT
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i])
            theta_dot_list.append(theta_dot)

        beta_dot_list,V,_,_,_,_ = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        state_dot = np.concatenate(theta_dot_list + beta_dot_list)
        return state_dot, V

    def get_state_dot_closed(self,t,state,V_base,A_base,BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        tau_list = [np.zeros(link.joint.nw) for link in self.links]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #ROTATIONS AND CONSTRAINT SETUPS
        link1 = self.links[0]
        linkn = self.links[-1]

        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])

        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d, -d])

        # OLD CODE 
        # #OPERTATIONAL SPACE INERTIA
        # omega_diag, omega_n1, omega_1n = self.omega(theta_list,tau_bar,D,n)

        # #calculating block entires
        # Λ_11 = IR1 @ (link1.RBT.T @ omega_diag[1] @ link1.RBT) @IR1.T
        # Λ_nn = IRn @ (omega_diag[n] @ IRn.T)
        # Λ_n1 = IR1 @ (omega_n1 @ link1.RBT) @ IR1.T
        # Λ_1n = IR1 @ (link1.RBT.T @ omega_1n) @ IR1.T


        #nyt forsøg på noget ekstremt smart
        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        omega_col_n = self.get_omega_ij_col(i, theta_list, tau_bar, omega_diag,n)
        omega_nn = omega_diag[n]
        omega_11 = omega_diag[1]

        omega_n1 = self.get_omega_ij(n,1,theta_list,tau_bar,omega_diag,n) #should live in frame 1

        

        Λ_11 = IR1 @ (link1.RBT.T @ omega_11 @ link1.RBT) @IR1.T
        Λ_nn = IRn @ (omega_nn @ IRn.T)
        Λ_n1 = IR1 @ (omega_n1 @ link1.RBT) @ IR1.T

        
    
        Λ_block = np.block([
            [Λ_11, Λ_n1.T],
            [Λ_n1, Λ_nn]
        ])

        V_tip = IR1@link1.RBT.T@V_f[1]
        v_tip  = V_tip[3:]
        v_base = IRn[:3, :3]@V_f[n][3:]

        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)

        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  -(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        #Φ_dot = -(IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge))
        Φ_dot =   v_tip - v_base
        Φ_ddot =  -(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        # Baumgarte stabilization
        α, β = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β)

        #solving for lagrange multipliers
        #λ = -np.linalg.solve((Q@Λ_block@Q.T),f)
        
        M_eff = Q @ Λ_block @ Q.T
        λ = -np.linalg.lstsq(M_eff, f, rcond=None)[0]
        print(f"{np.linalg.norm(Φ):.1e}")
        #calculating f_c
        f_c_closed_loop_const = -Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #constraints and Q are ordered [tip, base]

        f_c[1] = link1.RBT @ IR1.T @ f_c_closed_loop_const[:6] 
        f_c[n] = IRn.T @ f_c_closed_loop_const[6:]

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        if self._record_metrics == True:
            self.constraint_violation.append(np.linalg.norm(Φ))#for storage, can be uncommented if not in use


        return state_dot, V_f


    def get_state_dot_sprockets(self,t,state,V_base,A_base,BG_params,Penalty_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forced are usedto simulate damping
        damping = 0.1
        tau_list = [-damping * beta for beta in beta_list]
    

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)


        #positons and rotations. Needed for constraints and penalty
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        IR_list = self.get_all_rotations_body_to_I(theta_list)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC) - WITH PENALTY!
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI_penalty(t,theta_list,beta_list,tau_list,V_base,A_base,positions,IR_list,Penalty_params=Penalty_params)

        #ROTATIONS AND CONSTRAINT SETUPS
        link1 = self.links[0]
        linkn = self.links[-1]

        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])

        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d, -d])


        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        omega_nn = omega_diag[n]
        omega_11 = omega_diag[1]
        
        omega_n1 = self.get_omega_ij(n,1,theta_list,tau_bar,omega_diag,n)

        Λ_11 = IR1 @ (link1.RBT.T @ omega_11 @ link1.RBT) @IR1.T
        Λ_nn = IRn @ (omega_nn @ IRn.T)
        Λ_n1 = IR1 @ (omega_n1 @ link1.RBT) @ IR1.T

        
    
        Λ_block = np.block([
            [Λ_11, Λ_n1.T],
            [Λ_n1, Λ_nn]
        ])

        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  -(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        Φ_dot = -(IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge))
        Φ_ddot =  -(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        # Baumgarte stabilization
        alpha, beta = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, alpha, beta)

        #solving for lagrange multipliers
        #λ = -np.linalg.solve((Q@Λ_block@Q.T),f)
        
        λ = -np.linalg.solve((Q @ Λ_block @ Q.T), f)
        #calculating f_c
        f_c_closed_loop_const = -Q.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #constraints and Q are ordered [tip, base]

        f_c[1] = link1.RBT @ IR1.T @ f_c_closed_loop_const[:6] 
        f_c[n] = IRn.T @ f_c_closed_loop_const[6:]

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def get_state_dot_multiple_constraints(self,t,state,V_base,A_base,BG_params):
        #Previous implementation was not physical, it did not account for the cross-coupling between constrints and simply solved them independently. This is not correct. 

        #to test multiple constraints a n=3 body pendulum will be implemented. The two constraints are
            #1. Closed loop constraint between 1-3
            #2. A driving constraint on k=2
                #important note: This is not efficient at all, as the simulation scales with n_c³.
                #The resulting lambda matrix will be 6*n_c X 6_n_c - that is it will be 18x18. Similarly, we will have to stack Q matrices and Lambda matrices. Q will be a 6x18 matrix.
        

        #unpacking state and getting
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #compute positions of all 3 links
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        l_IO1 = positions[1]
        l_IO2 = positions[2]
        l_IOn = positions[n]
        

        #compute needed rotations
        #rotations. To keep general for now, we simply use notation that n=3. 
        linkn = self.links[-1]
        link1 = self.links[0]
        link2 = self.links[1]
        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IR2 = SOA.get_rotation_body_to_I(theta_list,self.links,n,2)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])
        

        #constraint 1 setup - Closed Loop between 1 and 3
        Q_closed = np.block([np.zeros((3,3)), np.eye(3),np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),-np.eye(3)]) #how to set this up check my paper handwriting. I think it could be a good idea to draw this in as an example.

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])


        Φ_closed =  -1*(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        Φ_closed_dot = -1*(IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge))
        Φ_closed_ddot =  -1*(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        #constraint 2 setup - Driver on link 2. in x-z plane

        Q_driver = np.block([np.zeros((3,3)), np.zeros((3,3)),np.zeros((3,3)),np.eye(3),np.zeros((3,3)),np.zeros((3,3))])

        IR2_3 = IR2[:3,:3]

        omega = np.pi
        length = np.linalg.norm(l_IO2 - l_IO1)

        l_driver = np.array([length*np.cos(omega*t),0,length*np.sin(omega*t)])
        l_driver_dot = np.array([-omega*length*np.sin(omega*t),0,omega*length*np.cos(omega*t)])
        l_driver_ddot = np.array([-omega**2*length*np.cos(omega*t),0,-omega**2*length*np.sin(omega*t)])

        Φ_driver = l_IO2  - l_driver
        Φ_driver_dot = IR2_3@V_f[2][3:] - l_driver_dot
        Φ_driver_ddot = IR2_3@A_f[2][3:]- l_driver_ddot

        #Calculating operational spatial space compliance entires. Needed are Ω(1,1), Ω(2,2), Ω(3,3), Ω(2,1), Ω(3,1), Ω(3,2). 
        #In this computation, we actually end up building the full Ω matrix using the omega function - this is not generally needed when more links are added, as the non-needed entries are then simply not calculated :)

        #calculation of Ω(1,1), Ω(2,2), Ω(3,3)
        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        Ω_11 = omega_diag[1]
        Ω_22 = omega_diag[2]
        Ω_33 = omega_diag[n]

       

        Ω_21 = self.get_omega_ij(2,1,theta_list,tau_bar,omega_diag,n)
        Ω_31 = self.get_omega_ij(3,1,theta_list,tau_bar,omega_diag,n)
        Ω_32 = self.get_omega_ij(3,2,theta_list,tau_bar,omega_diag,n)


        #time to build lambda matrix. Block entires are calculated
        #constraint 1 - closed loop
        Λ_11 = IR1 @ (link1.RBT.T @ Ω_11 @ link1.RBT) @IR1.T
        Λ_33 = IRn @ (Ω_33 @ IRn.T)
        Λ_31 = IR1 @ (Ω_31 @ link1.RBT) @ IR1.T
        Λ_13 = IR1 @ (link1.RBT.T @ Ω_31.T) @ IR1.T

        #constraint 2 - driver. Driving the base of the link here
        Λ_22 = IR2 @ (Ω_22 @ IR2.T)

        #cross couplings
        Λ_21 = IR1 @ (Ω_21 @ link1.RBT) @ IR1.T
        Λ_12 = IR1 @ (link1.RBT.T @ Ω_21.T) @ IR1.T
        Λ_32 = IR2 @ (Ω_32 @ IR2.T)
        Λ_23 = IR2 @ (Ω_32.T @ IR2.T)

        #assembling system quantities

        Λ_sys = np.block([
            [Λ_11, Λ_12, Λ_13],
            [Λ_21, Λ_22, Λ_23],
            [Λ_31, Λ_32, Λ_33]
        ])

        Q_sys = np.block([
            [Q_closed],
            [Q_driver]
        ])
        
        Φ_system = np.concatenate([Φ_closed, Φ_driver])

        Φ_dot_system = np.concatenate([Φ_closed_dot, Φ_driver_dot])
        Φ_ddot_system = np.concatenate([Φ_closed_ddot, Φ_driver_ddot])

        #Baumgarte stabilization    

        α, β = BG_params
        Φ_BG = SOA.baumgarte_stab(Φ_system, Φ_dot_system, Φ_ddot_system, α, β)
        print(np.linalg.norm(Φ_system))
        #solving for lagrange multipliers
        
        M_eff = Q_sys @ Λ_sys @ Q_sys.T
        λ = -np.linalg.solve(M_eff, Φ_BG)
    

        #calculating f_c
        f_const = -Q_sys.T@λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #constraints and Q are ordered [tip, base]
        f_c[1] = link1.RBT @ IR1.T @ f_const[:6]
        f_c[2] = IR2.T @ f_const[6:12]
        f_c[n] = IRn.T @ f_const[12:]

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f
    
    def run_ATBI(self,theta_list,beta_list,tau_list,V_base,A_base):
        n = len(self.links)

        theta = [None]*(n+2)
        beta  = [None]*(n+2)
        tau   = [None]*(n+2)
        links = [None]*(n+2) 

        for i in range(1, n+1):
            theta[i] = theta_list[i-1]
            beta[i]  = beta_list[i-1]
            tau[i]   = tau_list[i-1]
            links[i] = self.links[i-1]

        theta[0]   = np.zeros_like(theta[1])
        theta[n+1] = np.zeros_like(theta[n])
        beta[0]    = np.zeros_like(beta[1])
        beta[n+1]  = np.zeros_like(beta[n])
        tau[0]     = np.zeros_like(tau[1])
        tau[n+1]   = np.zeros_like(tau[n])

        self.l_from_origin = links[n].joint.get_translation(theta[n]) #returns [0,0,0] for anything other than FreeJoint()

        P_plus, xi_plus, nu, A, V, G, D, beta_dot, tau_bar, agothic, bgothic,pRc_cache = \
            [([None]*(n+2)) for _ in range(12)] 
            
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = np.zeros((6,6))
            
        A[n+1] = A_base
        V[n+1] = V_base
    
        # --- ATBI scatter ---- 
        for k in range(n, 0, -1):
            if k == n:
                RBT = SOA.RBT(self.l_from_origin)
            else:
                RBT = links[k+1].RBT

            pRc = links[k].joint.get_spatial_rotation(theta[k]) 
            pRc_cache[k] = pRc
            cRp = pRc.T 

            delta_V_k = links[k].joint.H.T @ beta[k]
            V[k] = cRp @ RBT.T @ V[k+1] + delta_V_k #k+1 as we need phi(k+1,k)

            agothic[k] = SOA.spatialskewtilde(V[k]) @ delta_V_k - SOA.spatialskewbar(delta_V_k)@delta_V_k
            bgothic[k] = SOA.spatialskewbar(V[k]) @ links[k].M @ V[k]

        # --- ATBI GATHER ---
        for k in range(1, n+1): 
            if k == 1:
                pRc = np.eye(6)
                cRp = pRc.T
            else:
                pRc = pRc_cache[k-1]
                cRp = pRc.T 

            P = links[k].RBT @ pRc @ P_plus[k-1] @ cRp @ links[k].RBT.T + links[k].M
            D[k] = links[k].joint.H @ P @ links[k].joint.H.T
            G[k] = np.linalg.solve(D[k], links[k].joint.H @ P).T 
            tau_bar[k] = np.eye(6) - G[k] @ links[k].joint.H
            P_plus[k] = tau_bar[k] @ P
            xi = links[k].RBT @ pRc @ xi_plus[k-1] + P @ agothic[k] + bgothic[k]
                    
            eps = tau[k] - links[k].joint.H @ xi
            nu[k] = np.linalg.solve(D[k], eps) 
            xi_plus[k] = xi + G[k] @ eps

        # --- 4. ATBI SCATTER ---
        for k in range(n, 0, -1):
            if k == n: #boundary condition on n. This is to model free joint if needed.
                RBT = SOA.RBT(self.l_from_origin )
            else:
                RBT = links[k+1].RBT

            pRc = pRc_cache[k]
            cRp = pRc.T 

            A_plus = cRp @ RBT.T @ A[k+1]
            beta_dot[k] = nu[k] - G[k].T @ A_plus
            A[k] = A_plus + links[k].joint.H.T @ beta_dot[k] + agothic[k]
        return beta_dot[1:n+1], V, A, tau_bar, D, G
    
    def run_ATBI_penalty(self,t,theta_list,beta_list,tau_list,V_base,A_base,positions,IR_list,Penalty_params):
        #nice to have = sprockets as input
        n = len(self.links)

        theta = [None]*(n+2)
        beta  = [None]*(n+2)
        tau   = [None]*(n+2)
        links = [None]*(n+2) 

        for i in range(1, n+1):
            theta[i] = theta_list[i-1]
            beta[i]  = beta_list[i-1]
            tau[i]   = tau_list[i-1]
            links[i] = self.links[i-1]

        theta[0]   = np.zeros_like(theta[1])
        theta[n+1] = np.zeros_like(theta[n])
        beta[0]    = np.zeros_like(beta[1])
        beta[n+1]  = np.zeros_like(beta[n])
        tau[0]     = np.zeros_like(tau[1])
        tau[n+1]   = np.zeros_like(tau[n])


        P_plus, xi_plus, nu, A, V, G, D, beta_dot, tau_bar, agothic, bgothic,pRc_cache = \
            [([None]*(n+2)) for _ in range(12)] 
            
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = np.zeros((6,6))
            
        A[n+1] = A_base
        V[n+1] = V_base
    
        # --- ATBI scatter (Kinematics) ---- 
        for k in range(n, 0, -1):
            pRc = links[k].joint.get_spatial_rotation(theta[k]) 
            pRc_cache[k] = pRc
            cRp = pRc.T 

            delta_V = links[k].joint.H.T @ beta[k]
            V[k] = cRp @ links[k].RBT.T @ V[k+1] + delta_V

            agothic[k] = SOA.spatialskewtilde(V[k]) @ links[k].joint.H.T @ beta[k]
            bgothic[k] = SOA.spatialskewbar(V[k]) @ links[k].M @ V[k]
        
        # --- PENALTY DETECTION --- #
        
        #intializing external force array. This could contain any external forces, but for now, its purely used for the forces coming from sprockets
        f_ext_body = [np.zeros(6,) for _ in range(n+2)]

        # Unpack stiffness and damping
        k_stiffness = Penalty_params[0]
        c_damping = Penalty_params[1]

        # ---- 5. GEOMETRY ----
        sprockets = [
             {'center': np.array([-1.0-(0.23*min(t,10)), 0.0, 0.0]), 'radius': 2.12}, # Left Sprocket
             {'center': np.array([ 3.3, 0.0, 0.0]), 'radius': 2.12}  # Right Sprocket
         ]
        
        #gammelt center 'center': np.array([-1.0-(0.23*min(t,10)), 0.0, 0.0]),
        #gammel radius var 2.125
        
        for k in range(1,n+1):
            IR_k = IR_list[k]  
            IR_k_3 = IR_k[:3, :3]
            pos = positions[k] #get current position of base of link k
            base_vel = IR_k_3 @ V[k][3:] #get current velocity

            #loop over sprockets. Right now there are two, but more can be added, thus a for loop is implemented
            for sprocket in sprockets:
                #calculating vector from sprocket center to current location aswell as distance
                vec_from_sprocket_center = pos - sprocket['center']
                distance = np.linalg.norm(vec_from_sprocket_center)

                # distance between body and spocket radius. If this is negative, then we have penetration
                d = distance - sprocket['radius']

                if d < 0: # Penetration into the sprocket (also a little cheating on the driving)

                    #geometry
                    normal_vec = vec_from_sprocket_center / distance #normal vec - this is based on where the link is at the time, and NOT where it was during penetration
                    #there is an argument for this being slightly inaccurate, but with a small enough dt the discreptency is expected to be rather small.
                    
                    tangent_vec = np.array([-normal_vec[2], 0.0, normal_vec[0]])
                    d_dot = np.dot(normal_vec, base_vel)
                    v_tangent = np.dot(tangent_vec,base_vel)
                    
                    if t>15 and sprocket['center'][0]>0:
                        F_drive_mag = 20.0
                    else: 
                        F_drive_mag = 0.0
                    
                    # Calculate pure spring compliant force and damping, pushing outward
                    F_normal_mag = -k_stiffness * d - c_damping * d_dot
                    if F_normal_mag < 0:
                        F_normal_mag = 0.0

                
                    # normal force
                    F_sprocket_3_out = F_normal_mag * normal_vec
                    #driving force
                    F_sprocket_3_drive = F_drive_mag*tangent_vec
                    # Transform force to body frame
                    F_body = IR_k_3.T @ (F_sprocket_3_out + F_sprocket_3_drive)

                    # add to body k. This also in theory should handle the case that more than 1 sprocket is hit.
                    f_ext_body[k][3:] += F_body


        # --- ATBI GATHER --- Now with external forces 
        for k in range(1, n+1): 
            if k == 1:
                pRc = np.eye(6)
                cRp = pRc.T
            else:
                pRc = pRc_cache[k-1]
                cRp = pRc.T 

            P = links[k].RBT @ pRc @ P_plus[k-1] @ cRp @ links[k].RBT.T + links[k].M
            D[k] = links[k].joint.H @ P @ links[k].joint.H.T
            G[k] = np.linalg.solve(D[k], links[k].joint.H @ P).T 
            tau_bar[k] = np.eye(6) - G[k] @ links[k].joint.H
            P_plus[k] = tau_bar[k] @ P
            # We incorporate our spatial penalty forces here as in algorithm from Jain (might need a bit of explenaton in the paper)
            xi = links[k].RBT @ pRc @ xi_plus[k-1] + P @ agothic[k] + bgothic[k] - f_ext_body[k] 
                    
            eps = tau[k] - links[k].joint.H @ xi
            nu[k] = np.linalg.solve(D[k], eps) 
            xi_plus[k] = xi + G[k] @ eps

        # --- 4. ATBI SCATTER ---
        for k in range(n, 0, -1):
            pRc = pRc_cache[k]
            cRp = pRc.T 

            A_plus = cRp @ links[k].RBT.T @ A[k+1]
            beta_dot[k] = nu[k] - G[k].T @ A_plus
            A[k] = A_plus + links[k].joint.H.T @ beta_dot[k] + agothic[k]
        return beta_dot[1:n+1], V, A, tau_bar, D, G
    
    def simulate(self, tspan, V_base, A_base, config="open", BG_params=None,Penalty_params=None):
        print(f"Simulation started ({config}-loop configuration)")
        start_time = time.perf_counter()

        # Initial configuration
        state0 = self.get_initial_state()
        dt = tspan[1] - tspan[0]
        
        nt = len(tspan)
        nq = len(state0)

        Y = np.zeros((nq, nt))
        Y[:, 0] = state0
        self.V = [None]*nt #to be able to save spatial velocities
        self.beta_dot = [None]*nt
        # Dynamically route the derivative calculation based on config
        def ODEfun(t, state, V_base, A_base):
            if config == "closed":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for closed-loop simulation.")
                return self.get_state_dot_closed(t, state, V_base, A_base, BG_params)
            elif config == "open":
                return self.get_state_dot(t, state, V_base, A_base)
            elif config == "driver":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for driver simulation.")
                return self.get_state_dot_driver(t, state, V_base, A_base, BG_params)
            elif config == "multiple_constraints":
                return self.get_state_dot_multiple_constraints(t, state, V_base, A_base, BG_params)
            elif config == "sprockets":
                if Penalty_params is None: # You can pass this through the BG_params argument or make a new one
                    raise ValueError("penalty_params (k, c) must be provided.")
                if BG_params is None:
                    raise ValueError("BG_params must be provided for sprockets simulation.")
                return self.get_state_dot_sprockets(t, state, V_base, A_base, BG_params, Penalty_params)
            else:
                raise ValueError("Invalid config. Choose 'open', 'closed' or 'driver'.")
        
        # RK4 integration loop
        for i in range(nt-1):
            t = tspan[i]
            y = Y[:, i]

            self._record_metrics = True
            k1, V_val  = ODEfun(t, y, V_base, A_base)
            self._record_metrics = False
            
            self.V[i] = V_val
            self.beta_dot[i] = k1[self.total_nq:]

            k2, _  = ODEfun(t + dt/2, y + dt/2 * k1, V_base, A_base)
            k3, _  = ODEfun(t + dt/2.0, y + dt/2.0 * k2, V_base, A_base)
            k4, _  = ODEfun(t + dt, y + dt * k3, V_base, A_base)

            Y[:, i+1] = y + dt/6.0 * (k1 + 2*k2 + 2*k3 + k4)

            # Robust way to print every 1 second of simulation time
            if t % 1 < dt: 
                print(f"t = {t:.2f} s")
        # Calc last V entry
        self._record_metrics = True
        state_dot_last, V_last = ODEfun(tspan[-1], Y[:,-1], V_base, A_base)
        self._record_metrics = False
        
        self.V[-1] = V_last
        self.beta_dot[-1] = state_dot_last[self.total_nq:]

        self.result = Y
        self.tspan = tspan
        end_time = time.perf_counter()
        elapesed_time = end_time - start_time
        print(f"Simulation finished. Runtime: {elapesed_time:.2f} s")
    
    def get_omega_diag(self,theta_list,tau_bar,D,n):
                #storage
        gamma = [None]*(n+2)
        omega = [None]*(n+2)
        theta = [None]*(n+2)
        links = [None]*(n+2) 


        for i in range(1,n+1):
            theta[i] = theta_list[i-1]
            links[i] = self.links[i-1]

        #boundary condition on omega
        gamma[n+1] = np.zeros((6,6))

        for k in range (n,0,-1):

            if k == n: #boundary condition on n. This is to model free joint if needed.
                RBT = SOA.RBT(self.l_from_origin )
            else:
                RBT = links[k+1].RBT

            #rotations
            pRc = links[k].joint.get_spatial_rotation(theta[k])
            cRp = pRc.T

            #calculating diagonal entries of omega

            gamma[k] = tau_bar[k].T @ cRp @ RBT.T @ gamma[k+1] @ RBT @ pRc @ tau_bar[k] + links[k].joint.H.T @ np.linalg.solve(D[k],links[k].joint.H)

        #renaminmg for readability
        omega_diag = gamma

        return omega_diag

    def get_omega_ij(self, i, j, theta_list, tau_bar, omega_diag,n):
        #calculates off diagonal entries not the MOST efficent as this may recalculate some entires, so essentially we are making more function calls than nessecarry. 
        #THIS IS NOT ORDER N FOR ANYTHING MORE THAN A SINGULAR CONSTRAINT! - IN THAT CASE, MAKE ANOTHER FUNCTION THAT RETURNS AND ENTIRE LIST
        if i == j:
            return omega_diag[i]
        
        if i < j:
            # Omega is symmetric: Omega_{i, j} = Omega_{j, i}^T
            return self.get_omega_ij(j, i, theta_list, tau_bar, omega_diag,n).T
            
        current_omega = omega_diag[i]
        theta = [None]*(n+2)
        links = [None]*(n+2) 

        for idx in range(1,n+1):
            theta[idx] = theta_list[idx-1]
            links[idx] = self.links[idx-1]
            
        # Propagate from body i-1 down to j
        for k in range(i-1, j-1, -1):
            pRc = links[k].joint.get_spatial_rotation(theta[k])
            cRp = pRc.T
            if k == n: #boundary condition on n. This is to model free joint if needed.
                RBT = SOA.RBT(self.l_from_origin )
            else:
                RBT = links[k+1].RBT

            current_omega = cRp @ current_omega @ RBT @ pRc @ tau_bar[k]
        #det den roterer lever i frame j    
        return current_omega

    def get_omega_ij_col(self, i, theta_list, tau_bar, omega_diag,n):
            #calculates entire col instead of a single entry, thus is returns a column. This scales O(N*n_b) due to the fact, that it has to be called as many times as there are nodes of interest. i.e n_b = no. of nodes
            #this is not a scaling issue for the closed loop, as it is only a single constraints. For efficient modelling of more than one constraint, maybe one should reconsider recoding this.
            #It is however, more efficient than the default get_omega_ij, as it has better scaling.
            #it returns a list thats indexed [omega(i,1), omega(i,2)...
            #det den roterer lever i frame j 
                

            #IMPORTANT! IT ONLY PROPAGATES DOWNWARDS, SO YOU NEED TO HAVE i>j always.   
            current_omega = omega_diag[i]
            omega_col = [None]*(n+2)
            
            # Save the diagonal entry
            omega_col[i] = current_omega
            
            # Shift theta to 1-based indexing for convenience
            theta = [None]*(len(self.links)+2)

            for idx in range(1,n+1):
                theta[idx] = theta_list[idx-1]
                
            for k in range(i - 1, 0, -1):
                link_k = self.links[k-1]
                pRc = link_k.joint.get_spatial_rotation(theta[k])
                cRp = pRc.T
  
                # Propagate one step down
                current_omega = cRp @ current_omega @ link_k.RBT @ pRc @ tau_bar[k]
            
                # Save the result, because this IS Omega_{i, k}. It should live in frame k
                omega_col[k] = current_omega
            return omega_col
    
    def beta_dot_delta(self,theta_list,tau_bar,D,f_c,G,n):
        #shifting indexing for convience (same method as in run_ATBI)
        n = len(self.links) #no of bodies

        theta = [None]*(n+2)
        links = [None]*(n+2) 
        
        for i in range(1, n+1):
            theta[i] = theta_list[i-1]
            links[i] = self.links[i-1]

        theta[0]   = np.zeros_like(theta[1])
        theta[n+1] = np.zeros_like(theta[n])
        links[0] = links[1] # For initialization, but doesn't matter, bc tau_bar is all zeros

        #storage
        xi_delta = [None]*(n+2)
        beta_dot_delta = [None] * (n+2)
        nu = [None]*(n+2)
        lambda_list = [None]*(n+2)

        #boundary conditions f xi_delta and lambda_list
        xi_delta[0] = np.zeros(6,)
        lambda_list[n+1] = np.zeros(6,)

        #gather pass
        for k in range(1,n+1):
            #rotations
            pRc = links[k-1].joint.get_spatial_rotation(theta[k-1])
            cRp = pRc.T

            xi_delta[k] = links[k].RBT @ pRc @ tau_bar[k-1] @ xi_delta[k-1] - f_c[k]
            
            nu[k] = np.linalg.solve(D[k],links[k].joint.H @ xi_delta[k])

        #scatter pass
        for k in range(n,0,-1):
            if k == n: #boundary condition on n. This is to model free joint if needed.
                RBT = SOA.RBT(self.l_from_origin )
            else:
                RBT = links[k+1].RBT
            #rotations
            pRc = links[k].joint.get_spatial_rotation(theta[k])
            cRp = pRc.T
            
            lambda_list[k] = tau_bar[k].T @ cRp @ RBT.T @ lambda_list[k+1]+links[k].joint.H.T@nu[k]

            beta_dot_delta[k] = nu[k] - G[k].T@cRp@RBT.T@lambda_list[k+1]
        
        return beta_dot_delta[1:n+1] #returning on 0 based indexing so it mathes

    def plot_gen_velocities(self):
        #renskrevet af gemini, havde problemer med nogle axer ifh til free joint plot

        n = len(self.links)
        idx = self.total_nq # Index where velocities start

        # Create figure for subplots
        fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)

        if n == 1:
            axes = [axes] 

        # Your standard RGB colors for X, Y, Z
        colors = ['#B22222', "#336933", '#000080']

        for k in range(n):
            link = self.links[k]
            nw = link.joint.nw 

            beta_k = self.result[idx:idx + nw, :] 
            tspan = self.tspan 

            ax_left = axes[k]

            # --- 6-DOF FREE JOINT ---
            if nw == 6: 
                ax_right = ax_left.twinx() # Create the independent right-hand axis

                # Angular (Left Axis, Solid Lines)
                ax_left.plot(tspan, beta_k[0, :], color=colors[0], linestyle='-', label=r'$\omega_x$')
                ax_left.plot(tspan, beta_k[1, :], color=colors[1], linestyle='-', label=r'$\omega_y$')
                ax_left.plot(tspan, beta_k[2, :], color=colors[2], linestyle='-', label=r'$\omega_z$')

                # Linear (Right Axis, Dashed Lines)
                ax_right.plot(tspan, beta_k[3, :], color=colors[0], linestyle='--', label=r'$v_x$')
                ax_right.plot(tspan, beta_k[4, :], color=colors[1], linestyle='--', label=r'$v_y$')
                ax_right.plot(tspan, beta_k[5, :], color=colors[2], linestyle='--', label=r'$v_z$')

                # Independent Labels and Legends
                ax_left.set_ylabel(f'Body {k+1} Ang\n[rad/s]', fontweight='bold')
                ax_right.set_ylabel(f'Body {k+1} Lin\n[m/s]', fontweight='bold', rotation=270, labelpad=15)
                
                ax_left.legend(loc='upper left', fontsize='small')
                ax_right.legend(loc='upper right', fontsize='small')

            # --- 3-DOF SPHERICAL JOINT ---
            elif nw == 3: 
                ax_left.plot(tspan, beta_k[0, :], color=colors[0], label=r'$\omega_x$')
                ax_left.plot(tspan, beta_k[1, :], color=colors[1], label=r'$\omega_y$')
                ax_left.plot(tspan, beta_k[2, :], color=colors[2], label=r'$\omega_z$')
                
                ax_left.set_ylabel(f'Body {k+1}\n[rad/s]')
                ax_left.legend(loc='upper right', fontsize='small')

            # --- 1-DOF REVOLUTE JOINT ---
            elif nw == 1: 
                if self.links[k].joint.axis == "x":
                    ax_left.plot(tspan, beta_k[0, :], color=colors[0], label=r'$\omega_x$')
                elif self.links[k].joint.axis == "y":
                    ax_left.plot(tspan, beta_k[0, :], color=colors[1], label=r'$\omega_y$')
                elif self.links[k].joint.axis == "z":
                    ax_left.plot(tspan, beta_k[0, :], color=colors[2], label=r'$\omega_z$')

                ax_left.set_ylabel(f'Body {k+1}\n[rad/s]')
                ax_left.legend(loc='upper right', fontsize='small')

            # Update index to start at the next body's data
            idx += nw

            # Shared grid settings
            ax_left.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time [s]')
        fig.suptitle('Generalized Velocities per Link', fontsize=14)
        plt.tight_layout()
        plt.show()

    def animation(self, config="openclosed", step=1):
        assert self.result is not None, "No simulation result found. Please run simulation before calling animation()."

        ani_tspan = self.tspan[::step]
        ani_states = self.result[:, ::step]

        # Number of bodies and time steps
        n = len(self.links)
        N = ani_states.shape[1]

        # Setting up the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Defining plotlim based on the number of bodies, link length and configuration
        total_link_length = sum(np.linalg.norm(link.l_hinge) for link in self.links)
        if config == "open":
            plotlim = total_link_length + np.linalg.norm(self.links[0].l_hinge)
        elif config == "closed":
            plotlim = total_link_length/2 + np.linalg.norm(self.links[0].l_hinge)
        else:
            raise ValueError("Invalid config value. Use 'open' or 'closed'.")
        
        # Set plot limits and labels
        ax.set_xlim([-plotlim, plotlim])
        ax.set_ylim([-plotlim, plotlim])
        ax.set_zlim([-plotlim, plotlim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set(box_aspect=(1, 1, 1))

        # Initialize the line object that will be updated in the animation
        line, = ax.plot([], [], [], 'o-', lw=2)

        def compute_positions(state_k):
            theta_list ,_ = self.unpack_state(state_k)
            positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
            
            # positions is constructed as a list of arrays, where the index 0 is empty. We can conveniently insert the tip of the last link as the first element in the list,
            # so we have all positions in one array.
            R3_tip2I = SOA.get_rotation_tip_to_body_I(theta_list, self.links, n)[:3,:3]
            #R3_tip2I = R6_tip2I
            tip_pos = (positions[1] + R3_tip2I @ self.links[0].l_hinge).flatten()
            positions[0] = tip_pos
            
            return np.array(positions) # Convert list of arrays to a single 2D array of shape (n_bodies+1, 3) for easier plotting
        
        # Update function for animation
        def update(frame):
            state_k = ani_states[:, frame]
            positions = compute_positions(state_k)
            line.set_data(positions[:, 0], positions[:, 1])
            line.set_3d_properties(positions[:, 2])
            ax.set_title(f"t = {ani_tspan[frame]:.3f} s")

        # Just a robust way of calculating the interval between frames for the animation, based on the time vector. Could also do tspan[1] - tspan[0]. 
        dt = np.mean(np.diff(ani_tspan))
        interval = dt * 1000 # Convert to milliseconds for FuncAnimation
        ani = FuncAnimation(
            fig, update, frames=N, interval=interval, blit=False
        )
        
        ax.view_init(elev=0, azim=-90, roll=0)
        plt.show()
        return ani

    def plot_initial_state(self, config="openclosed"):
        # Number of bodies and time steps
        n = len(self.links)

        # Setting up the figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Defining plotlim based on the number of bodies, link length and configuration
        total_link_length = sum(np.linalg.norm(link.l_hinge) for link in self.links)
        if config == "open":
            plotlim = total_link_length + np.linalg.norm(self.links[0].l_hinge)
        elif config == "closed":
            plotlim = total_link_length/2 + np.linalg.norm(self.links[0].l_hinge)
        else:
            raise ValueError("Invalid config value. Use 'open' or 'closed'.")
        
        # Set plot limits and labels
        ax.set_xlim([-plotlim, plotlim])
        ax.set_ylim([-plotlim, plotlim])
        ax.set_zlim([-plotlim, plotlim])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set(box_aspect=(1, 1, 1))

        state0 = self.get_initial_state()
        theta_list ,_ = self.unpack_state(state0)
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        # positions is constructed as a list of arrays, where the index 0 is empty. We can conveniently insert the tip of the last link as the first element in the list,
        # so we have all positions in one array.
        R3_tip2I = SOA.get_rotation_tip_to_body_I(theta_list, self.links, n)[:3,:3]
        #R3_tip2I = R6_tip2I
        tip_pos = (positions[1] + R3_tip2I @ self.links[0].l_hinge).flatten()
        positions[0] = tip_pos
        pos_array = np.array(positions)     
       
        # Plotting the "skeleton"
        ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2], 'o-', lw=3, markersize=8)

        ax.view_init(elev=0, azim=-90, roll=0)
        plt.grid(True)
        plt.show()

        return fig, ax

    def compute_com_pos_in_inertial_frame(self, theta):
        #OBS! SKAL ÆNDRES LÆNGERE OPPE SÅ VI IKKE LÆNGERE BRUGER SOA.PY
        n = len(self.links)
        
        positions = [None]*(n+1)
        com_positions = [None]*(n+1)

        R_cumulative = SOA.rotfromquat(theta[n-1]) #initial rotation from body n to inertial frame

        #BC for position of base body
        positions[n] = np.zeros(3)
        com_positions[n] = R_cumulative @ self.links[n-1].l_com

        for i in range(n-1,0,-1):
            pRc = SOA.rotfromquat(theta[i-1]) # bc theta starts from 0

            positions[i] = positions[i+1] + R_cumulative @ self.links[i].l_hinge # self.links start from 0...
            
            R_cumulative = R_cumulative @ pRc

            com_positions[i] = positions[i] + R_cumulative @ self.links[i-1].l_com # self.links start from 0...

        return com_positions

    def calc_energies(self, z0):
        # Colab between Kap and Gemini
        """
        Calculates the kinetic, potential, and total energy of the system for all time steps.
        Saves the results as 1D numpy arrays in self.KE, self.PE, and self.TE.
        
        Args:
            z0: A scalar or list of length n, specifying the potential energy reference offset for each body.
        """

        n = len(self.links)
        
        if self.result is None:
            raise ValueError("Simulation must be run before calculating energies.")
            
        if len(z0) != n:
            raise ValueError(f"z0 must be of length {n} (one offset per link)")
        
        # Adjust for indexing
        z0 = np.insert(z0, 0, 0)

        nt = len(self.tspan)
        self.KE = np.zeros(nt)
        self.PE = np.zeros(nt)
        self.TE = np.zeros(nt)
        
        g = 9.81
        
        for i in range(nt):
            # Initalization for each timestep
            KE_t = 0.0
            PE_t = 0.0

            # Current state
            state = self.result[:, i]
            theta_list, _ = self.unpack_state(state)            
            
            # Compute com positions of hinges in the inertial frame
            com_pos = self.compute_com_pos_in_inertial_frame(theta_list)

            for k in range(n, 0, -1):
                link = self.links[k-1]
                
                # Kinetic Energy for this link (0.5 * V.T * M * V)
                Vk = self.V[i][k]
                KE_t += 0.5 * (Vk.T @ link.M @ Vk)
                
                # Potential Energy for this link (m * g * h)
                zk_pot = com_pos[k][-1] + z0[k]  # z-coordinate + offset
                PE_t += link.m * g * zk_pot
                    
            self.KE[i] = KE_t
            self.PE[i] = PE_t
            self.TE[i] = KE_t + PE_t
        
        print("Energies calculated!")

    def CSV_creator(self, path, filename, *attr_names):
        # Made mainly by Gemini
        """
        Merges an arbitrary number of attributes (lists/arrays stored in self) 
        into a CSV file, where each attribute represents a column. 
        Raises a ValueError if the attributes do not have the same number of rows.
        """
        
        if not attr_names:
            print("No attribute names provided.")
            return
            
        extracted_lists = []
        for name in attr_names:
            if not hasattr(self, name):
                raise AttributeError(f"The system does not have an attribute named '{name}'.")
            
            attr_data = getattr(self, name)
            
            # Handle lists that might contain mixed elements (e.g., None and arrays)
            if isinstance(attr_data, list):
                processed_data = []
                for row in attr_data:
                    if isinstance(row, (list, tuple, np.ndarray)):
                        flat_row = []
                        for item in row:
                            if item is None:
                                continue
                            elif isinstance(item, (int, float, str, np.number)):
                                flat_row.append(item)
                            else:
                                flat_row.extend(np.asarray(item).flatten().tolist())
                        processed_data.append(flat_row)
                    else:
                        processed_data.append(row)
                arr = np.asarray(processed_data)
            else:
                arr = np.asarray(attr_data)
                
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            extracted_lists.append(arr)
            
        expected_length = extracted_lists[0].shape[0]
        
        for i in range(len(extracted_lists)):
            if extracted_lists[i].shape[0] != expected_length:
                # Automatically transpose wide arrays (like self.result) to match expected row count
                if extracted_lists[i].ndim == 2 and extracted_lists[i].shape[1] == expected_length:
                    extracted_lists[i] = extracted_lists[i].T
                else:
                    raise ValueError(f"Attribute '{attr_names[i]}' has {extracted_lists[i].shape[0]} rows, expected {expected_length}.")
                
        if not filename.endswith('.csv'):
            filename += '.csv'
            
        # Combine arrays horizontally to support multiple columns per array
        combined_data = np.hstack(extracted_lists)
        
        # Combine path and filename
        path_filename = path + "/" + filename
        
        with open(path_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(combined_data)
            
        print(f"Data successfully saved as {filename} in {path}.")
      
    def get_state_dot_wall_penalty(self, t, state, V_base, A_base, penalty_params):
        """
        Wall contact using a Compliant Penalty Method (Spring-Damper).
        penalty_params = (k_stiffness, c_damping)
        """
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        # 1. Unconstrained Dynamics (Gravity, Coriolis, internal damping)
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]

        theta_dot_list = []
        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i], beta_list[i])
            theta_dot_list.append(theta_dot)

        # Run ATBI to get free accelerations and spatial velocities (V_f)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list, beta_list, tau_list, V_base, A_base)

        # 2. Get global positions
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        # Initialize f_c with zeros for all bodies. This will hold our spatial penalty forces.
        f_c = [np.zeros(6,) for _ in range(n+2)]
        
        k_stiffness, c_damping = penalty_params
        
        # DEFINE YOUR WALL HERE
        wall_pos = np.array([-0.0, 0.0, 0.0])
        wall_normal = np.array([1.0, 0.0, 0.0]) # Must be normalized!
        
        contact_detected = False

        # 3. Check for collisions and accumulate penalty forces
        for i in range(1, n+1):
            link = self.links[i-1]
            IR_i = SOA.get_rotation_body_to_I(theta_list, self.links, n, i)
            IR_i_3 = IR_i[:3, :3]
            
            # Tip kinematics in global frame
            tip_pos = positions[i] + IR_i_3 @ link.l_hinge
            omega_tilde_i = SOA.skewfromvec(IR_i_3 @ V_f[i][:3])
            tip_vel = IR_i_3 @ V_f[i][3:] + omega_tilde_i @ IR_i_3 @ link.l_hinge
            
            # Distance and velocity along the normal
            phi = np.dot((tip_pos - wall_pos), wall_normal)
            phi_dot = np.dot(tip_vel, wall_normal)
            
            # If penetrated, apply spring-damper force
            if phi < 0:
                contact_detected = True
                
                # Penalty Force Model: Spring (k) + Damper (c)
                # Max(0, ...) ensures the wall only PUSHES, never pulls the pendulum in.
                F_normal_mag = max(0.0, -k_stiffness * phi - c_damping * phi_dot)
                
                # Convert to 3D Force vector in Inertial Frame
                F_I = F_normal_mag * wall_normal
                
                # Spatial Wrench at the tip in Inertial Frame [Torque; Force]
                W_tip_I = np.concatenate([np.zeros(3), F_I])
                
                # Transform wrench to body hinge frame and accumulate.
                # NOTE: We use -W_tip_I because your beta_dot_delta method naturally subtracts f_c.
                # By passing negative, the minus signs cancel and we push the pendulum OUT of the wall.
                f_c_body = link.RBT @ IR_i.T @ (-W_tip_I)
                f_c[i] = f_c[i] + f_c_body

        # 4. Map spatial forces to joint accelerations using your existing ATBI delta method
        if contact_detected:
            beta_dot_delta_list = self.beta_dot_delta(theta_list, tau_bar, D, f_c, G, n)
            beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]
        else:
            beta_dot_final_list = beta_dot_f_list

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def calc_TE_delta(self):
        if self.result is None:
            raise ValueError("Simulation must be run before calculating energies.")
        
        # if self.TE_delta is not None:
        #     raise ValueError("calc_energies has already been run.")
        
        n = len(self.links)
        nt = len(self.tspan)
        self.TE_delta = np.zeros(nt)

        g = 9.81
        
        for i in range(nt):
            # Initalization for each timestep
            KE_rel_t = 0.0
            PE_rel_t = 0.0

            # Current state
            state = self.result[:, i]
            theta_list, _ = self.unpack_state(state)            
            
            # Compute com positions of hinges in the inertial frame
            com_pos = self.compute_com_pos_in_inertial_frame(theta_list)
            
            if i == 0: # Initial instance
                com_pos_ini = com_pos
                Vk_ini = self.V[i]

            for k in range(n, 0, -1):
                link = self.links[k-1]
                
                # Kinetic Energy for this link
                Vk = self.V[i][k]
                KE_t = 0.5 * (Vk.T @ link.M @ Vk)
                KE_ini = 0.5 * (Vk_ini[k].T @ link.M @ Vk_ini[k])
                KE_rel_t += KE_t - KE_ini

                # Relative potential Energy for this link
                zk_pot_rel = com_pos[k][-1] - com_pos_ini[k][-1]
                PE_rel_t += link.m * g * zk_pot_rel

            self.TE_delta[i] = KE_rel_t + PE_rel_t

        print("TE_delta calculated!")

    def calc_and_plot_penetration(self):
        """
        Calculates and plots the maximum penetration depth of any joint into the sprockets over time.
        """
        if self.result is None:
            raise ValueError("Simulation must be run before calculating penetration.")
            
        n = len(self.links)
        nt = len(self.tspan)
        max_penetrations = np.zeros(nt)
        
        for i in range(nt):
            t = self.tspan[i]
            state = self.result[:, i]
            theta_list, _ = self.unpack_state(state)
            positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
            
            sprockets = (
                 (np.array([-1.0-(0.23*min(t,10)), 0.0, 0.0]) , 2.12), # Left Sprocket
                 (np.array([ 3.3, 0.0, 0.0]), 2.12)                      # Right Sprocket
             )
            #gammelt center np.array([-1.0-(0.23*min(t, 10.0))

            max_pen = 0.0
            for k in range(1, n+1):
                pos = positions[k]
                for center, radius in sprockets:
                    dist = np.linalg.norm(pos - center)
                    pen = radius - dist
                    if pen > max_pen:
                        max_pen = pen
            max_penetrations[i] = max_pen
            
        self.penetration = max_penetrations
        
        plt.figure(figsize=(10, 5))
        plt.plot(self.tspan, max_penetrations * 1000, color='red', label='Max Penetration')
        plt.xlabel('Time [s]')
        plt.ylabel('Penetration Depth [mm]')
        plt.title('Maximum Joint Penetration into Sprockets over Time')
        plt.grid(True)
        plt.legend()
        plt.show()

    def get_all_rotations_body_to_I(self, theta_list):
        """
        Computes the spatial rotation matrix from each body's frame to the inertial frame 
        in a single O(n) sweep. Returns a 1-indexed list of 6x6 spatial rotation matrices.
        """
        n = len(self.links)
        IR_list = [None] * (n + 1)
        
        # Start at the base (body n)
        # Note: self.links and theta_list are 0-indexed, so body n is at index n-1
        IR_cumulative = self.links[n-1].joint.get_spatial_rotation(theta_list[n-1])
        IR_list[n] = IR_cumulative
        
        # Sweep down the chain from n-1 to the tip (1)
        for k in range(n-1, 0, -1):
            pRc = self.links[k-1].joint.get_spatial_rotation(theta_list[k-1])
            IR_cumulative = IR_cumulative @ pRc  # Multiply parent's rotation by child's relative rotation
            IR_list[k] = IR_cumulative
            
        return IR_list