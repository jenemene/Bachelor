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

        tau_list = [np.zeros(link.joint.nw) for link in self.links]

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
        
        λ = -np.linalg.solve((Q @ Λ_block @ Q.T), f)
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

        return state_dot, V_f

    def get_state_dot_driver(self,t,state,V_base,A_base,BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        #tau_list = [np.zeros(link.joint.nw) for link in self.links]
        damping = 0.1
        tau_list = [-damping * beta for beta in beta_list]

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

        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])

        Q = np.block([np.zeros((3,3)), np.eye(3)])

        #OPERTATIONAL SPACE INERTIA
        omega_diag, _, _ = self.omega(theta_list,tau_bar,D,n)

        # DRIVER
        #calculating block entires
        Λ_nn = IRn @ (omega_diag[n] @ IRn.T)

        Λ_block = Λ_nn

        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        l_IOn = positions[n]

        r = 0.2
        ω = np.pi #angular velocity of the driver
        center = np.array([0,0,0])
        bias = 0
        driver, driver_dot, driver_ddot = self.circle_driver_xz_plane(r, t, ω, center, bias)

        x_on = 1
        Φ = l_IOn - driver
        Φ_dot = IRn[:3, :3]@V_f[n][3:] - driver_dot
        Φ_ddot = IRn[:3, :3]@A_f[n][3:] - driver_ddot
        
        #x_on = 1
        #Φ = l_IOn - np.array([x_on*0.2*np.cos(ω*t), 0, 0.2*np.sin(ω*t)]) #driver is moving in a circle in the xz plane
        #Φ_dot = IRn[:3, :3]@V_f[n][3:]  - np.array([-x_on*ω*0.2*np.sin(ω*t), 0, ω*0.2*np.cos(ω*t)])
        #Φ_ddot = IRn[:3, :3]@A_f[n][3:] - np.array([-x_on*ω**2*0.2*np.cos(ω*t), 0, -ω**2*0.2*np.sin(ω*t)])

        # Baumgarte stabilization
        α, β = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β)

        #solving for lagrange multipliers
        λ = -np.linalg.lstsq((Q @ Λ_block @ Q.T), f, rcond=None)[0]

        #calculating f_c
        f_c_closed_loop_const = -Q.T @ λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        f_c[n] = IRn.T @ f_c_closed_loop_const





        # HOLDING CONSTRAINT
        #calculating block entires
        IR1 = SOA.get_rotation_tip_to_body_I(theta_list, self.links, n)
        Λ_11 = IR1 @ (link1.RBT.T @ omega_diag[1] @ link1.RBT ) @ IR1.T

        Λ_block_h = Λ_11

        l_IO1 = positions[1]
        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])

        Φ_h = (l_IO1 + IR1[:3, :3]@link1.l_hinge) - np.array([0.4, 0, 0])
        Φ_dot_h = (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge)
        Φ_ddot_h = (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge)

        # Baumgarte stabilization
        
        f_h = SOA.baumgarte_stab(Φ_h, Φ_dot_h, Φ_ddot_h, α, β)

        #solving for lagrange multipliers
        λ_h = -np.linalg.lstsq((Q @ Λ_block_h @ Q.T), f_h, rcond=None)[0]

        #calculating f_c
        f_c_closed_loop_const_h = -Q.T @ λ_h

        f_c[1] = link1.RBT @ IR1.T @ f_c_closed_loop_const_h

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        Φ_norm = np.linalg.norm(Φ)
        Φ_norm_h = np.linalg.norm(Φ_h)
        #print(f"Time = {t:.2f}   Driver = {Φ_norm:.2e}    Constraint = {Φ_norm_h:.2e}")
        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def get_state_dot_driver_pentagon(self,t,state,V_base,A_base,BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        #tau_list = [np.zeros(link.joint.nw) for link in self.links]
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #Q matrix (only constraints on translation, not rotation)
        Q = np.block([np.zeros((3,3)), np.eye(3)])

        #OPERTATIONAL SPACE INERTIA
        omega_diag, omega_n1, omega_1n = self.omega(theta_list,tau_bar,D,n)

        #radius for a circle with a pentagon inscribed
        #assuming all links are equal length. Formula from googling: "pentagon inscribed in a circle formula"
        division_term = np.sqrt((5-np.sqrt(5))/2)
        r = np.linalg.norm(self.links[0].l_hinge) / division_term

        #other params
        ω = np.pi #angular velocity of the driver
        center = np.array([0,0,0])
        bias = 0 #starting bias

        #construction of f_c vector
        f_c = [np.zeros(6,) for _ in range(n+2)]

        #positions of all joints in inertial frame
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        #for holding constraint violations
        Φ_circle = [None]*(n+2)
        
        IRi = np.eye(6) # To initialize rotation scatter

        #for loop to constrain bodies to circle motion using bilaterial constraints
        for i in range(n, 0, -1): #starts at body n, goes down until, and including, body 2. Body 1 will be constrained later.
            IRi = IRi @ self.links[i-1].joint.get_spatial_rotation(theta_list[i-1]) #[i-1] because self.links and theta_list starts from 0

            #calling driver func to get constraint terms
            driver, driver_dot, driver_ddot = self.circle_driver_xz_plane(r, t, ω, center, bias)

            #i'th position in inertial frame
            l_IOi = positions[i]

            # constraints and derivs.
            Φ_circle[i] = l_IOi - driver
            Φ_dot = IRi[:3, :3]@V_f[i][3:] - driver_dot
            Φ_ddot = IRi[:3, :3]@A_f[i][3:] - driver_ddot

            # Baumgarte stabilization
            α, β = BG_params
            f = SOA.baumgarte_stab(Φ_circle[i], Φ_dot, Φ_ddot, α, β)

            #big lambda
            Λ_i = IRi @ omega_diag[i] @ IRi.T

            #solving for lagrange multipliers
            λ = -np.linalg.lstsq((Q @ Λ_i @ Q.T), f, rcond=None)[0]

            #calculating f_c and inserting into f_c
            f_c_closed_loop_const = -Q.T @ λ
            f_c[i] = IRi.T @ f_c_closed_loop_const
            
            #updating bias for next loop
            bias = bias - 2*np.pi/5 #from Gemini, see "https://gemini.google.com/share/6e4f72c34dd2"

        



        #CONSTRAINT BETWEEN BODY 1 AND BODY N (copied from "get_state_dot_closed")
        #ROTATIONS AND CONSTRAINT SETUPS
        link1 = self.links[0]
        linkn = self.links[-1]

        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IRn = linkn.joint.get_spatial_rotation(theta_list[-1])

        d = np.block([np.zeros((3,3)), np.eye(3)])
        Q = np.block([d, -d])

        #calculating block entires
        Λ_11 = IR1 @ (link1.RBT.T @ omega_diag[1] @ link1.RBT) @IR1.T
        Λ_nn = IRn @ (omega_diag[n] @ IRn.T)
        Λ_n1 = IR1 @ (omega_n1 @ link1.RBT) @ IR1.T
        Λ_1n = IR1 @ (link1.RBT.T @ omega_1n) @ IR1.T

        Λ_block = np.block([
            [Λ_nn, Λ_n1.T],
            [Λ_1n.T, Λ_11]
        ])

        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge)
        Φ_dot = IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge)
        Φ_ddot =  IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge)

        # Baumgarte stabilization
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β)
        
        λ = -np.linalg.lstsq((Q @ Λ_block @ Q.T), f, rcond=None)[0]

        #calculating f_c and adding to previous values not to overwrite 
        f_c_closed_loop_const = -Q.T@λ
        f_c[n] = f_c[n] + IRn.T @ f_c_closed_loop_const[:6]
        f_c[1] = f_c[1] + link1.RBT @ IR1.T @ f_c_closed_loop_const[6:] 
        
        Φ_5 = np.linalg.norm(Φ_circle[5])
        Φ_4 = np.linalg.norm(Φ_circle[4])
        Φ_3 = np.linalg.norm(Φ_circle[3])
        Φ_2 = np.linalg.norm(Φ_circle[2])
        Φ_1 = np.linalg.norm(Φ_circle[1])

        print(f"t = {t:.3f}     Φ_5 = {Φ_5:.2e}  Φ_4 = {Φ_4:.2e}  Φ_3 = {Φ_3:.2e}  Φ_2 = {Φ_2:.2e}  Φ_1 = {Φ_1:.2e}  Φ_51 = {np.linalg.norm(Φ):.2e}")

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def get_state_dot_driver_bottom(self,t,state,V_base,A_base,BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        tau_list = [np.zeros(link.joint.nw) for link in self.links]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i])
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #ROTATIONS AND CONSTRAINT SETUPS
        link1 = self.links[0]

        IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)

        Q = np.block([np.zeros((3,3)), np.eye(3)])

        #OPERTATIONAL SPACE INERTIA
        omega_diag, _, _ = self.omega(theta_list,tau_bar,D,n)

        #calculating block entires
        Λ_11 = IR1 @ (link1.RBT.T @ omega_diag[1] @ link1.RBT) @IR1.T

        Λ_block = Λ_11

        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        l_IO1 = positions[1]
        IR1_3 = IR1[:3,:3]
        ω_tilde_I = SOA.skewfromvec(IR1_3 @ V_f[1][:3])

        ω = np.pi

        f_driver = np.array([0.2 + 0.2*np.sin(ω*t),0,0])
        f_d_driver = np.array([0.2 - 0.2*ω*np.cos(ω*t),0,0])
        f_dd_driver = np.array([0.2 - 0.2*ω**2*np.sin(ω*t),0,0])

        Φ = l_IO1 + IR1_3@link1.l_hinge - f_driver
        Φ_dot = IR1_3@V_f[1][3:] + ω_tilde_I@IR1_3@link1.l_hinge - f_d_driver
        Φ_ddot = IR1_3@A_f[1][3:] + SOA.skewfromvec(IR1_3@A_f[1][:3])@IR1_3@link1.l_hinge + ω_tilde_I@ω_tilde_I@IR1_3@link1.l_hinge - f_dd_driver

        print(f"t={t:.2f}  |Φ| = {np.linalg.norm(Φ):.2e}")

        # Baumgarte stabilization
        α, β = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β)

        #solving for lagrange multipliers
        #λ = -np.linalg.lstsq((Q @ Λ_block @ Q.T), f, rcond=None)[0]
        λ = -np.linalg.solve(Q@Λ_block@Q.T,f) # Dimension: 3x1

        #calculating f_c
        f_c_closed_loop_const = -Q.T @ λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        f_c[1] = link1.RBT @ IR1.T @ f_c_closed_loop_const

        #calculating beta_dot_delta
        beta_dot_delta_list = self.beta_dot_delta(theta_list,tau_bar,D,f_c,G,n)

        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]

        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

    def get_state_dot_unilateral_constraints(self,t,state,V_base,A_base):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        #tau_list = [np.zeros(link.joint.nw) for link in self.links]
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #compute positions
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)

        link = self.links[n-1]
        l_IO1 = positions[n]
        #IR1 = SOA.get_rotation_tip_to_body_I(theta_list,self.links,n)
        IR1 = link.joint.get_spatial_rotation(theta_list[-1])
        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[n][:3])
    
        Φ_f = (l_IO1 + IR1[:3, :3]@link.l_hinge)
        Φ_f = Φ_f[0] + 0.1
        Φ_dot_f = (IR1[:3, :3]@V_f[n][3:] + IωIO@IR1[:3, :3]@link.l_hinge)
        Φ_dot_f = Φ_dot_f[0]
        Φ_ddot_f = (IR1[:3, :3]@A_f[n][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[n][:3])@IR1[:3, :3]@link.l_hinge + IωIO@IωIO@IR1[:3,:3]@link.l_hinge)
        Φ_ddot_f = Φ_ddot_f[0]

        #Q matrix (only constraints on z)
        Q = np.array([0,0,0,1,0,0]).reshape(1,6)

        #OPERTATIONAL SPACE INERTIA
        omega_diag, _, _ = self.omega(theta_list,tau_bar,D,n)
        Λ_11 = IR1 @ (link.RBT.T @ omega_diag[n] @ link.RBT) @IR1.T

        # checking for active state
        # ADD LATER
        o = 1
        if o == 1:
            if Φ_f <= 0 and Φ_ddot_f <= 0:
                M = Q @ Λ_11 @ Q.T
                d = Φ_ddot_f

                kp = 1000.0  # Position gain
                kd = 1000.0   # Velocity gain
                
                # Modified 'd' to account for penetration and approach velocity
                d_stabilized = Φ_ddot_f + kd * Φ_dot_f + kp * Φ_f
                λ = -d_stabilized/M

                # lam = cp.Variable(1)
                # prob = cp.Problem(cp.Minimize(0.5 * lam * M * lam + d * lam),[lam >= 0])
                # prob.solve()
                # λ = lam.value
            else:
                λ = np.array([0])
        else:
            M = Q @ Λ_11 @ Q.T
            d = Φ_ddot_f

            kp = 1000.0  # Position gain
            kd = 1000.0   # Velocity gain
            
            # Modified 'd' to account for penetration and approach velocity
            d_stabilized = Φ_ddot_f + kd * Φ_dot_f + kp * Φ_f
            λ = -d_stabilized/M
        
        #print(f"t = {t:.2f}     Φ = {Φ_f:.2e}   Φ_ddot_f = {Φ_ddot_f:.2e}   M = {M}     λ = {λ}")

        f_c = [np.zeros(6,) for _ in range(n+2)]
        f_c_closed_loop_const = -Q.T@λ
        f_c[n] = link.RBT @ IR1.T @ f_c_closed_loop_const
        f_c[n] = f_c[n].flatten()
        #print(f_c_closed_loop_const)

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

        #calculation og off diagonal terms <--- HVIS DER ER EN FEJL SÅ START HER BED OMEGA UDREGNINGERNE (jeg har debugget de virker lowkey)
        Ω_21 = self.get_omega_ij(2, 1, theta_list, tau_bar, omega_diag,n)
        Ω_31 = self.get_omega_ij(3, 1, theta_list, tau_bar, omega_diag,n)
        Ω_32 = self.get_omega_ij(3, 2, theta_list, tau_bar, omega_diag,n) #jeg vil jo mene at 32 bliver regnet for at regne 31, så man skal måske bare retunere en liste med dem ned af

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


    def get_state_dot_driver_debug(self,t,state,V_base,A_base,BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forces (set to 0 for now, could be used if wanted)
        tau_list = [np.zeros(link.joint.nw) for link in self.links]

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i])
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC)
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list,beta_list,tau_list,V_base,A_base)

        #ROTATIONS AND CONSTRAINT SETUPS
        link2 = self.links[1]

        IR2 = SOA.get_rotation_body_to_I(theta_list,self.links,n,2)

        Q = np.block([np.zeros((3,3)), np.eye(3)])

        #OPERTATIONAL SPACE INERTIA
        omega_diag = self.get_omega_diag(theta_list,tau_bar,D,n)
        omega_22 = omega_diag[2]

        #calculating block entires
        Λ_22 = IR2 @ (omega_22) @IR2.T

        Λ_block = Λ_22

        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        
        l_IO2 = positions[2]
        l_IO1 = positions[1]
        IR2_3 = IR2[:3,:3]
        ω_tilde_I = SOA.skewfromvec(IR2_3 @ V_f[2][:3])

        omega = np.pi
        length = np.linalg.norm(l_IO2 - l_IO1)

        l_driver = np.array([length*np.cos(omega*t),0,length*np.sin(omega*t)])
        l_driver_dot = np.array([-omega*length*np.sin(omega*t),0,omega*length*np.cos(omega*t)])
        l_driver_ddot = np.array([-omega**2*length*np.cos(omega*t),0,-omega**2*length*np.sin(omega*t)])

        Φ = l_IO2  - l_driver
        Φ_dot = IR2_3@V_f[2][3:] - l_driver_dot
        Φ_ddot = IR2_3@A_f[2][3:]- l_driver_ddot

        print(f"t={t:.2f}  |Φ| = {np.linalg.norm(Φ):.2e}")

        # Baumgarte stabilization
        α, β = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, α, β)

        #solving for lagrange multipliers
        λ = -np.linalg.lstsq((Q @ Λ_block @ Q.T), f, rcond=None)[0]
        #λ = -np.linalg.solve(Q@Λ_block@Q.T,f) # Dimension: 3x1

        #calculating f_c
        f_c_closed_loop_const = -Q.T @ λ
        f_c = [np.zeros(6,) for _ in range(n+2)]

        f_c[2] =  IR2.T @ f_c_closed_loop_const

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

        P_plus, xi_plus, nu, A, V, G, D, beta_dot, tau_bar, agothic, bgothic = \
            [([None]*(n+2)) for _ in range(11)] 
            
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = np.zeros((6,6))
            
        A[n+1] = A_base
        V[n+1] = V_base
    
        # --- ATBI scatter ---- 
        for k in range(n, 0, -1):
            pRc = links[k].joint.get_spatial_rotation(theta[k]) 
            cRp = pRc.T 

            delta_V = links[k].joint.H.T @ beta[k]
            V[k] = cRp @ links[k].RBT.T @ V[k+1] + delta_V

            agothic[k] = SOA.spatialskewtilde(V[k]) @ links[k].joint.H.T @ beta[k]
            bgothic[k] = SOA.spatialskewbar(V[k]) @ links[k].M @ V[k]

        # --- ATBI GATHER ---
        for k in range(1, n+1): 
            if k == 1:
                pRc = np.eye(6)
                cRp = pRc.T
            else:
                pRc = links[k-1].joint.get_spatial_rotation(theta[k-1])
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
            pRc = links[k].joint.get_spatial_rotation(theta[k])
            cRp = pRc.T 

            A_plus = cRp @ links[k].RBT.T @ A[k+1]
            beta_dot[k] = nu[k] - G[k].T @ A_plus
            A[k] = A_plus + links[k].joint.H.T @ beta_dot[k] + agothic[k]
        return beta_dot[1:n+1], V, A, tau_bar, D, G
    
    def run_ATBI_penalty(self,t,theta_list,beta_list,tau_list,V_base,A_base,Penalty_params):
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

        P_plus, xi_plus, nu, A, V, G, D, beta_dot, tau_bar, agothic, bgothic = \
            [([None]*(n+2)) for _ in range(11)] 
            
        P_plus[0] = np.zeros((6,6))
        xi_plus[0] = np.zeros((6,))
        tau_bar[0] = np.zeros((6,6))
            
        A[n+1] = A_base
        V[n+1] = V_base
    
        # --- ATBI scatter (Kinematics) ---- 
        for k in range(n, 0, -1):
            pRc = links[k].joint.get_spatial_rotation(theta[k]) 
            cRp = pRc.T 

            delta_V = links[k].joint.H.T @ beta[k]
            V[k] = cRp @ links[k].RBT.T @ V[k+1] + delta_V

            agothic[k] = SOA.spatialskewtilde(V[k]) @ links[k].joint.H.T @ beta[k]
            bgothic[k] = SOA.spatialskewbar(V[k]) @ links[k].M @ V[k]
        
        # --- PENALTY DETECTION --- #
        positions = SOA.compute_pos_in_inertial_frame(theta_list,self.links,n) #computing positions
        
        #intializing external force array. This could contain any external forces, but for now, its purely used for the forces coming from sprockets
        f_ext_body = [np.zeros(6,) for _ in range(n+2)]

        # Unpack just the stiffness from the parameters
        k_stiffness = Penalty_params[0]

        #sprocket geometry and position
        sprockets = [
            {'center': np.array([-1.0-(0.28*min(t,10)), 0.0, 0.0]), 'radius': 2.1}, # Left Sprocket
            {'center': np.array([ 3.3, 0.0, 0.0]), 'radius': 2.1}  # Right Sprocket
        ]
        for k in range(1,n+1):
            IR_k = SOA.get_rotation_body_to_I(theta_list, self.links, n, k) #getting rotation matrix. Needed to rotate penalty force into body frame later
            IR_k_3 = IR_k[:3, :3]
            pos = positions[k] #get current position of base of link k

            #loop over sprockets. Right now there are two, but more can be added, thus a for loop is implemented
            for sprocket in sprockets:
                #calculating vector from sprocket center to current location aswell as distance
                vec_from_sprocket_center = pos - sprocket['center']
                distance = np.linalg.norm(vec_from_sprocket_center)

                # distance between body and spocket radius. If this is negative, then we have penetration
                d = distance - sprocket['radius']

                if d < 0: # Penetration into the sprocket (also a little cheating on the driving)
                    normal_vec = vec_from_sprocket_center / distance #normal vec - this is based on where the link is at the time, and NOT where it was during penetration
                    #there is an argument for this being slightly inaccurate, but with a small enough dt the discreptency is expected to be rather small.
                    
                    #calcuating the tangent vector to simulate the sprocket driving it around. We could optimize this and maybe implement friction and a rotating sprocket.
                    tangent_vec = np.array([-normal_vec[1], normal_vec[0], 0.0])
                    F_drive_mag = 50.0

                    # Calculate pure spring compliant force, pushing outward
                    F_normal_mag = -k_stiffness * d
                    
                    # 3D force vector in inertial frame, purely along the normal
                    F_sprocket_3_out = F_normal_mag * normal_vec
                    
                    F_sprocket_3_drive = F_drive_mag*tangent_vec
                    # Transform force to body frame
                    F_body = IR_k_3.T @ (F_sprocket_3_drive + F_sprocket_3_drive)

                    # add to body k. This also in theory should handle the case that more than 1 sprocket is hit.
                    f_ext_body[k] = f_ext_body[k] + np.concatenate([np.zeros(3), F_body])


        # --- ATBI GATHER --- Now with external forces 
        for k in range(1, n+1): 
            if k == 1:
                pRc = np.eye(6)
                cRp = pRc.T
            else:
                pRc = links[k-1].joint.get_spatial_rotation(theta[k-1])
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
            pRc = links[k].joint.get_spatial_rotation(theta[k])
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
            elif config == "pentagon":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for driver simulation.")
                return self.get_state_dot_driver_pentagon(t, state, V_base, A_base, BG_params)
            elif config == "driver_bottom":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for driver simulation.")
                return self.get_state_dot_driver_bottom(t, state, V_base, A_base, BG_params)
            elif config == "driver_debug":
                return self.get_state_dot_driver_debug(t, state, V_base, A_base, BG_params)
            elif config == "unilateral_constraints":
                return self.get_state_dot_unilateral_constraints(t, state, V_base, A_base)
            elif config == "multiple_constraints":
                return self.get_state_dot_multiple_constraints(t, state, V_base, A_base, BG_params)
            elif config == "wall_contact":
                if BG_params is None:
                    raise ValueError("BG_params must be provided for contact simulation.")
                return self.get_state_dot_wall_contact(t, state, V_base, A_base, BG_params)
            elif config == "wall_penalty":
                if Penalty_params is None: # You can pass this through the BG_params argument or make a new one
                    raise ValueError("penalty_params (k, c) must be provided.")
                return self.get_state_dot_wall_penalty(t, state, V_base, A_base, Penalty_params)
            elif config == "wall_penalty_CL":
                return self.get_state_dot_closed_penalty(t, state, V_base, A_base, BG_params, Penalty_params)
            else:
                raise ValueError("Invalid config. Choose 'open', 'closed' or 'driver'.")
        
        # RK4 integration loop
        for i in range(nt-1):
            t = tspan[i]
            y = Y[:, i]

            k1, V_val  = ODEfun(t, y, V_base, A_base)
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
        state_dot_last, V_last = ODEfun(tspan[-1], Y[:,-1], V_base, A_base)
        self.V[-1] = V_last
        self.beta_dot[-1] = state_dot_last[self.total_nq:]

        self.result = Y
        self.tspan = tspan
        end_time = time.perf_counter()
        elapesed_time = end_time - start_time
        print(f"Simulation finished. Runtime: {elapesed_time:.2f} s")

    def omega(self,theta_list,tau_bar,D,n):
        #storage
        gamma = [None]*(n+2)
        omega = [None]*(n+2)
        theta = [None]*(n+2)

        #theta_list is on a 0-index basis, for convenience i shift this. This is not effective in time, but for now is ok

        for i in range(1,n+1):
            theta[i] = theta_list[i-1]

        #boundary condition on omega
        gamma[n+1] = np.zeros((6,6))

        for k in range (n,0,-1):
            link_k = self.links[k-1] #remember, links is on a 0-index

            #rotations
            pRc = link_k.joint.get_spatial_rotation(theta[k])
            cRp = pRc.T

            #calculating diagonal entries of omega

            gamma[k] = tau_bar[k].T @ cRp @ link_k.RBT.T @ gamma[k+1] @ link_k.RBT @ pRc @ tau_bar[k] + link_k.joint.H.T @ np.linalg.solve(D[k],link_k.joint.H)

        #assigning them
        omega[n] = gamma[n]

        #calcualting off diagonal entries (and also inserting the one on the diagonal
        for k in range(n-1,0,-1):
            link_k = self.links[k-1] #remember, links is on a 0-index

            #rotations
            pRc = link_k.joint.get_spatial_rotation(theta[k])
            cRp = pRc.T

            omega[k] = cRp @ omega[k+1] @ link_k.RBT @ pRc @ tau_bar[k]
        
        #assigning calculated omegas
        omega_n1 = omega[1]
        omega_1n = omega_n1.T

        #all digonals of omega:
        omega_diag = gamma

        return omega_diag, omega_n1, omega_1n
    
    def get_omega_diag(self,theta_list,tau_bar,D,n):
                #storage
        gamma = [None]*(n+2)
        omega = [None]*(n+2)
        theta = [None]*(n+2)

        #theta_list is on a 0-index basis, for convenience i shift this. This is not effective in time, but for now is ok

        for i in range(1,n+1):
            theta[i] = theta_list[i-1]

        #boundary condition on omega
        gamma[n+1] = np.zeros((6,6))

        for k in range (n,0,-1):
            link_k = self.links[k-1] #remember, links is on a 0-index

            #rotations
            pRc = link_k.joint.get_spatial_rotation(theta[k])
            cRp = pRc.T

            #calculating diagonal entries of omega

            gamma[k] = tau_bar[k].T @ cRp @ link_k.RBT.T @ gamma[k+1] @ link_k.RBT @ pRc @ tau_bar[k] + link_k.joint.H.T @ np.linalg.solve(D[k],link_k.joint.H)

        #renaminmg for readability
        omega_diag = gamma

        return omega_diag

    def get_omega_ij(self, i, j, theta_list, tau_bar, omega_diag,n):
        #calculates off diagonal entries not the MOST efficent as this may recalculate some entires, so essentially we are making more function calls than nessecarry. I right now i cant think of a way to fix this, but i know there is one
        if i == j:
            return omega_diag[i]
        
        if i < j:
            # Omega is symmetric: Omega_{i, j} = Omega_{j, i}^T
            return self.get_omega_ij(j, i, theta_list, tau_bar, omega_diag,n).T
            
        current_omega = omega_diag[i]
        
        # Shift theta to 1-based indexing for convenience
        theta = [None]*(len(self.links)+2)

        for idx in range(1,n+1):
            theta[idx] = theta_list[idx-1]
            
        # Propagate from body i-1 down to j
        for k in range(i-1, j-1, -1):
            link_k = self.links[k-1]
            pRc = link_k.joint.get_spatial_rotation(theta[k])
            cRp = pRc.T
            current_omega = cRp @ current_omega @ link_k.RBT @ pRc @ tau_bar[k]
        #det den roterer lever i frame j    
        return current_omega
  
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
            #rotations
            pRc = links[k].joint.get_spatial_rotation(theta[k])
            cRp = pRc.T
            
            lambda_list[k] = tau_bar[k].T @ cRp @ links[k].RBT.T @ lambda_list[k+1]+links[k].joint.H.T@nu[k]

            beta_dot_delta[k] = nu[k] - G[k].T@cRp@links[k].RBT.T@lambda_list[k+1]
        
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

    def circle_driver_xz_plane(self, r, t, omega, center, bias):
        # Returns the driver components for the constraints equation
        # args:
        # r: radius of the circle
        # t: time
        # omega: angular velocity of the driver
        # center: center of the circle (array-like of shape (3,))
        # bias: the phase bias of the driver

        Φ = np.array([center[0] + r * np.cos(omega * t + bias), 0, center[2] + r * np.sin(omega * t + bias)])
        Φ_dot = np.array([-r * omega * np.sin(omega * t + bias), 0, r * omega * np.cos(omega * t + bias)])
        Φ_ddot = np.array([-r * omega**2 * np.cos(omega * t + bias), 0, -r * omega**2 * np.sin(omega * t + bias)])

        return Φ, Φ_dot, Φ_ddot

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
      
    def get_state_dot_wall_contact(self, t, state, V_base, A_base, BG_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        # 1. Unconstrained Dynamics 
        damping = 0.0
        tau_list = [-damping * beta for beta in beta_list]

        theta_dot_list = []
        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i], beta_list[i])
            theta_dot_list.append(theta_dot)

        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI(theta_list, beta_list, tau_list, V_base, A_base)

        # 2. Get global positions and Omega diagonals
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        omega_diag = self.get_omega_diag(theta_list, tau_bar, D, n)
        
# 3. THE SCANNER: Universal Contact Logic
        penetrating_indices = []
        contact_data = {}
        
        # DEFINE YOUR WALL HERE
        
        wall_pos = np.array([-0.0, 0.0, 0.0])
        wall_normal = np.array([1, 0.0, 0]) # Must be normalized!
        
        for i in range(1, n+1):
            link = self.links[i-1]
            IR_i = SOA.get_rotation_body_to_I(theta_list, self.links, n, i)
            IR_i_3 = IR_i[:3, :3]
            
            # Tip kinematics in global frame
            tip_pos = positions[i] + IR_i_3 @ link.l_hinge
            omega_tilde_i = SOA.skewfromvec(IR_i_3 @ V_f[i][:3])
            tip_vel = IR_i_3 @ V_f[i][3:] + omega_tilde_i @ IR_i_3 @ link.l_hinge
            tip_acc = IR_i_3 @ A_f[i][3:] + SOA.skewfromvec(IR_i_3 @ A_f[i][:3]) @ IR_i_3 @ link.l_hinge + omega_tilde_i @ omega_tilde_i @ IR_i_3 @ link.l_hinge
            
            # THE UNIVERSAL GAP FUNCTION
            # Dot product inherently handles all positive/negative directional logic!
            phi = np.dot((tip_pos - wall_pos), wall_normal)
            
            # If phi <= 0, the tip has crossed the boundary against the normal
            if phi <= 1e-4: 
                penetrating_indices.append(i)
                
                # Project velocities and accelerations exactly along the normal
                contact_data[i] = {
                    'IR': IR_i,
                    'phi': phi, 
                    'phi_dot': np.dot(tip_vel, wall_normal),
                    'phi_ddot': np.dot(tip_acc, wall_normal),
                    'link': link
                }

        # 4. THE ACTIVE SET METHOD (LCP Solver)
        active_set = list(penetrating_indices)
        optimal_lambda = None
        optimal_active_set = []
        Q_sys = None
        
        while True:
            n_c = len(active_set)
            
            if n_c == 0:
                optimal_lambda = np.zeros(0)
                optimal_active_set = []
                break
                
            Q_sys = np.zeros((n_c, 6 * n_c))
            Lambda_sys = np.zeros((6 * n_c, 6 * n_c))
            Phi_system = np.zeros(n_c)
            Phi_dot_system = np.zeros(n_c)
            Phi_ddot_system = np.zeros(n_c)
            
            for row, idx_i in enumerate(active_set):
                data_i = contact_data[idx_i]
                link_i = data_i['link']
                IR_i = data_i['IR']
                
                # The Q matrix strictly takes the normal vector components
                Q_sys[row, row * 6 + 3] = wall_normal[0]
                Q_sys[row, row * 6 + 4] = wall_normal[1]
                Q_sys[row, row * 6 + 5] = wall_normal[2]
                
                Phi_system[row] = data_i['phi']
                Phi_dot_system[row] = data_i['phi_dot']
                Phi_ddot_system[row] = data_i['phi_ddot']
                
                for col, idx_j in enumerate(active_set):
                    data_j = contact_data[idx_j]
                    link_j = data_j['link']
                    IR_j = data_j['IR']
                    
                    idx_min = min(idx_i, idx_j)
                    idx_max = max(idx_i, idx_j)
                    link_min = self.links[idx_min - 1]
                    link_max = self.links[idx_max - 1]
                    
                    # We ONLY need the IR of the minimum index, because Omega 
                    # has already rotated everything down into this frame!
                    IR_min = contact_data[idx_min]['IR'] 
                    
                    Omega_max_min = self.get_omega_ij(idx_max, idx_min, theta_list, tau_bar, omega_diag, n)
                    
                    if idx_i == idx_j:
                        # Diagonal: Everything lives in its own frame
                        Lambda_block = IR_i @ (link_i.RBT.T @ Omega_max_min @ link_i.RBT) @ IR_i.T
                        
                    elif idx_i > idx_j: 
                        # i is max, j is min
                        # Omega outputs in frame min. We strictly use IR_min to get to global!
                        Lambda_block = IR_min @ (link_max.RBT.T @ Omega_max_min @ link_min.RBT) @ IR_min.T
                        
                    else: 
                        # idx_i < idx_j
                        # i is min, j is max
                        # It is the mathematical transpose of the block above.
                        Lambda_block = IR_min @ (link_min.RBT.T @ Omega_max_min.T @ link_max.RBT) @ IR_min.T

                    Lambda_sys[row*6:(row+1)*6, col*6:(col+1)*6] = Lambda_block
                    
            alpha, beta_param = BG_params
            Phi_BG = SOA.baumgarte_stab(Phi_system, Phi_dot_system, Phi_ddot_system, alpha, beta_param)
            print(np.linalg.norm(Phi_system))
            M_eff = Q_sys @ Lambda_sys @ Q_sys.T

            print(M_eff.shape)
            
            lambda_forces = -np.linalg.solve(M_eff, Phi_BG)
            
            # PURE LCP CHECK: Because of the dot product, Lambda is universally positive!
            if np.all(lambda_forces >= -1e-10): 
                optimal_lambda = lambda_forces
                optimal_active_set = active_set
                break
            else:
                # If a force is negative, it's an impossible pulling force. Kick it out!
                min_idx = np.argmin(lambda_forces)
                active_set.pop(min_idx)
        # 5. Apply Output Forces to the Engine
        f_c = [np.zeros(6,) for _ in range(n+2)]
        
        if len(optimal_active_set) > 0:
            f_const = -Q_sys.T @ optimal_lambda
            for idx, body_idx in enumerate(optimal_active_set):
                link = contact_data[body_idx]['link']
                IR_i = contact_data[body_idx]['IR']
                
                # Extract the 6D global force vector and shift back to local COM
                f_body_global = f_const[idx*6 : (idx+1)*6]
                f_c[body_idx] = link.RBT @ IR_i.T @ f_body_global

        # 6. Correct and Return
        beta_dot_delta_list = self.beta_dot_delta(theta_list, tau_bar, D, f_c, G, n)
        beta_dot_final_list = [b_f + b_delta for b_f, b_delta in zip(beta_dot_f_list, beta_dot_delta_list)]
        state_dot = np.concatenate(theta_dot_list + beta_dot_final_list)

        return state_dot, V_f

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
    
    def get_state_dot_closed_penalty(self,t,state,V_base,A_base,BG_params,Penalty_params):
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)

        #generalized forced are usedto simulate damping
        damping = 20
        tau_list = [-damping * beta for beta in beta_list]
    

        #CALCULATION OF THETA_DOT
        theta_dot_list = []

        for i in range(len(self.links)):
            theta_dot = self.links[i].joint.get_derrivative(theta_list[i],beta_list[i]) #CAN CHANGE THIS TO PREALLOCATE FOR SPEED OPTIMIZATION!
            theta_dot_list.append(theta_dot)

        #UNCONSTRAINED FORWARD DYNAMICS (FREE VEL AND ACC) - WITH PENALTY!
        beta_dot_f_list, V_f, A_f, tau_bar, D, G = self.run_ATBI_penalty(t,theta_list,beta_list,tau_list,V_base,A_base,Penalty_params=Penalty_params)

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

        V_tip = IR1@link1.RBT.T@V_f[1]
        v_tip  = V_tip[3:]
        v_base = IRn[:3, :3]@V_f[n][3:]

        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n) #regnes faktisk i penalty også, så måske lidt overflødig

        l_IO1 = positions[1]
        l_IOn = positions[n]

        IωIO = SOA.skewfromvec(IR1[:3,:3]@V_f[1][:3])
    
        Φ =  -(l_IOn - (l_IO1 + IR1[:3, :3]@link1.l_hinge))
        #Φ_dot = -(IRn[:3, :3]@V_f[n][3:]  - (IR1[:3, :3]@V_f[1][3:] + IωIO@IR1[:3, :3]@link1.l_hinge))
        Φ_dot =   v_tip - v_base
        Φ_ddot =  -(IRn[:3, :3]@A_f[n][3:] - (IR1[:3, :3]@A_f[1][3:] + SOA.skewfromvec(IR1[:3, :3]@A_f[1][:3])@IR1[:3, :3]@link1.l_hinge + IωIO@IωIO@IR1[:3,:3]@link1.l_hinge))

        # Baumgarte stabilization
        alpha, beta = BG_params
        f = SOA.baumgarte_stab(Φ, Φ_dot, Φ_ddot, alpha, beta)

        #solving for lagrange multipliers
        #λ = -np.linalg.solve((Q@Λ_block@Q.T),f)
        
        λ = -np.linalg.solve((Q @ Λ_block @ Q.T), f)
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

        return state_dot, V_f

    def apply_wall_impulse(self, state):
        """
        Calculates and applies simultaneous impulses to ALL links that have 
        hit the wall at this exact microsecond, handling their cross-coupling.
        """
        theta_list, beta_list = self.unpack_state(state)
        n = len(self.links)
        
        # 1. Quick sweep to get pre-impact spatial velocities
        tau_list = [np.zeros(link.joint.nw) for link in self.links]
        _, V_f, _, tau_bar, D, G = self.run_ATBI(theta_list, beta_list, tau_list, np.zeros(6), np.zeros(6))
        
        positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, n)
        omega_diag = self.get_omega_diag(theta_list, tau_bar, D, n)
        
        wall_pos = np.array([-0.0, 0.0, 0.0])
        wall_normal = np.array([1.0, 0.0, 0.0])
        
        # 2. THE SCANNER: Find ALL links that are touching AND moving inward
        active_set = []
        contact_data = {}
        
        for i in range(1, n+1):
            link = self.links[i-1]
            IR_i = SOA.get_rotation_body_to_I(theta_list, self.links, n, i)
            IR_i_3 = IR_i[:3, :3]
            
            tip_pos = positions[i] + IR_i_3 @ link.l_hinge
            omega_tilde_i = SOA.skewfromvec(IR_i_3 @ V_f[i][:3])
            tip_vel = IR_i_3 @ V_f[i][3:] + omega_tilde_i @ IR_i_3 @ link.l_hinge
            
            phi = np.dot((tip_pos - wall_pos), wall_normal)
            phi_dot = np.dot(tip_vel, wall_normal)
            
            # If it's on the wall AND heading into the wall
            if phi <= 1e-4 and phi_dot < -1e-6: 
                active_set.append(i)
                contact_data[i] = {
                    'IR': IR_i,
                    'phi_dot': phi_dot,
                    'link': link
                }

        n_c = len(active_set)
        if n_c == 0:
            return state # Nothing to bounce!
            
        print(f"--- SIMULTANEOUS IMPACT RESOLVED: {n_c} links hit at once! ---")
            
        # 3. BUILD THE GLOBAL IMPACT MATRICES (Just like your LCP!)
        Q_sys = np.zeros((n_c, 6 * n_c))
        Lambda_sys = np.zeros((6 * n_c, 6 * n_c))
        Phi_dot_pre = np.zeros(n_c)
        
        for row, idx_i in enumerate(active_set):
            data_i = contact_data[idx_i]
            
            Q_sys[row, row * 6 + 3] = wall_normal[0]
            Q_sys[row, row * 6 + 4] = wall_normal[1]
            Q_sys[row, row * 6 + 5] = wall_normal[2]
            
            Phi_dot_pre[row] = data_i['phi_dot']
            
            for col, idx_j in enumerate(active_set):
                data_j = contact_data[idx_j]
                
                idx_min = min(idx_i, idx_j)
                idx_max = max(idx_i, idx_j)
                link_min = self.links[idx_min - 1]
                link_max = self.links[idx_max - 1]
                
                IR_min = contact_data[idx_min]['IR'] 
                Omega_max_min = self.get_omega_ij(idx_max, idx_min, theta_list, tau_bar, omega_diag, n)
                
                if idx_i == idx_j:
                    Lambda_block = data_i['IR'] @ (data_i['link'].RBT.T @ Omega_max_min @ data_i['link'].RBT) @ data_i['IR'].T
                elif idx_i > idx_j: 
                    Lambda_block = IR_min @ (link_max.RBT.T @ Omega_max_min @ link_min.RBT) @ IR_min.T
                else: 
                    Lambda_block = IR_min @ (link_min.RBT.T @ Omega_max_min.T @ link_max.RBT) @ IR_min.T

                Lambda_sys[row*6:(row+1)*6, col*6:(col+1)*6] = Lambda_block

        # 4. SOLVE THE GLOBAL IMPACT EQUATION
        e = 1.0  # Coefficient of Restitution
        M_eff = Q_sys @ Lambda_sys @ Q_sys.T
        
        # Delta V = -(1 + e) * V_pre
        delta_v_target = -(1.0 + e) * Phi_dot_pre
        
        # Solve for the vector of impulses!
        impulse_array = np.linalg.solve(M_eff, delta_v_target)
        
        # 5. DISTRIBUTE IMPULSES BACK TO THE SYSTEM
        f_impulse = [np.zeros(6,) for _ in range(n+2)]
        f_const_global = -Q_sys.T @ impulse_array
        
        for idx, body_idx in enumerate(active_set):
            link = contact_data[body_idx]['link']
            IR_i = contact_data[body_idx]['IR']
            
            f_body_global = f_const_global[idx*6 : (idx+1)*6]
            # No negative sign needed here because f_const_global already negated the Q projection
            f_impulse[body_idx] = link.RBT @ IR_i.T @ f_body_global
            
        # 6. UPDATE VELOCITIES
        delta_beta_impulse = self.beta_dot_delta(theta_list, tau_bar, D, f_impulse, G, n)
        new_beta_list = [b + db for b, db in zip(beta_list, delta_beta_impulse)]
        
        return np.concatenate(theta_list + new_beta_list)
    
    def simulate_scipy(self, tspan, V_base, A_base, config="open", BG_params=None, Penalty_params=None, method='RK45', **kwargs):
        """
        Simulates the multi-body system using scipy.integrate.solve_ivp.
        Includes Event-Driven root-finding for Rigid LCP contact (config="wall_contact").
        """
        print(f"Simulation started ({config}-loop configuration) using SciPy {method}")
        start_time = time.perf_counter()

        # Initial configuration
        state0 = self.get_initial_state()
        last_printed_t = [-1]

        # Dynamically route the derivative calculation based on config
        def ODEfun(t, state, V_base, A_base):
            if config == "closed":
                if BG_params is None: raise ValueError("BG_params must be provided.")
                return self.get_state_dot_closed(t, state, V_base, A_base, BG_params)
            elif config == "open":
                return self.get_state_dot(t, state, V_base, A_base)
            elif config == "driver":
                if BG_params is None: raise ValueError("BG_params must be provided.")
                return self.get_state_dot_driver(t, state, V_base, A_base, BG_params)
            elif config == "pentagon":
                if BG_params is None: raise ValueError("BG_params must be provided.")
                return self.get_state_dot_driver_pentagon(t, state, V_base, A_base, BG_params)
            elif config == "driver_bottom":
                if BG_params is None: raise ValueError("BG_params must be provided.")
                return self.get_state_dot_driver_bottom(t, state, V_base, A_base, BG_params)
            elif config == "driver_debug":
                return self.get_state_dot_driver_debug(t, state, V_base, A_base, BG_params)
            elif config == "unilateral_constraints":
                return self.get_state_dot_unilateral_constraints(t, state, V_base, A_base)
            elif config == "multiple_constraints":
                return self.get_state_dot_multiple_constraints(t, state, V_base, A_base, BG_params)
            elif config == "wall_contact":
                if BG_params is None: raise ValueError("BG_params must be provided.")
                return self.get_state_dot_wall_contact(t, state, V_base, A_base, BG_params)
            elif config == "wall_penalty":
                if Penalty_params is None: raise ValueError("Penalty_params must be provided.")
                return self.get_state_dot_wall_penalty(t, state, V_base, A_base, Penalty_params)
            elif config == "wall_penalty_CL":
                return self.get_state_dot_closed_penalty(t, state, V_base, A_base, BG_params, Penalty_params)
            else:
                raise ValueError("Invalid config.")

        # SciPy requires the derivative function to return ONLY dy/dt. 
        def scipy_fun(t, state):
            current_t = int(t)
            if current_t > last_printed_t[0]:
                print(f"Integration progress: t = {current_t} s / {int(tspan[-1])} s")
                last_printed_t[0] = current_t
                
            state_dot, _ = ODEfun(t, state, V_base, A_base)
            return state_dot

        # ==========================================
        # EVENT DETECTION SETUP (For Rigid LCP)
        # ==========================================
        def collision_event(t, state):
            theta_list, _ = self.unpack_state(state)
            positions = SOA.compute_pos_in_inertial_frame(theta_list, self.links, len(self.links))
            
            min_phi = 1.0 
            wall_pos = np.array([-0.0, 0.0, 0.0])
            wall_normal = np.array([1.0, 0.0, 0.0])
            
            for k in range(1, len(self.links)+1):
                link = self.links[k-1]
                IR_k = SOA.get_rotation_body_to_I(theta_list, self.links, len(self.links), k)[:3, :3]
                tip_pos = positions[k] + IR_k @ link.l_hinge
                
                phi = np.dot((tip_pos - wall_pos), wall_normal)
                if phi < min_phi:
                    min_phi = phi
            
            # Offset by a microscopic amount (1e-6) so we halt just before tunneling
            return min_phi - 1e-6 

        collision_event.terminal = True 
        collision_event.direction = -1  # Only trigger moving towards the wall
        
        events_list = [collision_event] if config == "wall_contact" else None

        # ==========================================
        # EVENT-DRIVEN INTEGRATION LOOP
        # ==========================================
        t_current = tspan[0]
        t_end = tspan[-1]
        current_state = state0
        
        all_t = []
        all_y = []

        while t_current < t_end:
            # Note: We specifically remove t_eval here so SciPy can use variable 
            # stepping to find the exact root of the impact equation.
            sol = solve_ivp(
                fun=scipy_fun,
                t_span=(t_current, t_end),
                y0=current_state,
                method=method,
                events=events_list,
                **kwargs
            )
            
            # Avoid appending exact duplicate boundary points
            if len(all_t) > 0:
                all_t.append(sol.t[1:])
                all_y.append(sol.y[:, 1:])
            else:
                all_t.append(sol.t)
                all_y.append(sol.y)
            
            if sol.status == 1: # Status 1 = Terminal Event Reached
                t_impact = sol.t[-1]
                state_at_impact = sol.y[:, -1]
                
                print(f"💥 Event triggered at t = {t_impact:.5f}s. Calculating Impulse...")
                
                # 1. APPLY IMPULSE: Snap the velocity to zero
                state_post_impact = self.apply_wall_impulse(state_at_impact)
                
                # 2. RESTART INTEGRATOR: Update tracking vars
                t_current = t_impact
                current_state = state_post_impact
                
            else:
                break # Reached t_end successfully

        # ==========================================
        # POST-PROCESSING & INTERPOLATION
        # ==========================================
        full_t = np.concatenate(all_t)
        full_y = np.hstack(all_y)
        
        # Interpolate the dense, variable-step data back onto the user's smooth tspan
        # This prevents animations and energy calculations from crashing
        from scipy.interpolate import interp1d
        interpolator = interp1d(full_t, full_y, axis=1, kind='linear', fill_value="extrapolate")
        
        self.tspan = tspan
        self.result = interpolator(tspan)
        nt = len(self.tspan)
        
        # Initialize storage for V and beta_dot
        self.V = [None] * nt
        self.beta_dot = [None] * nt
        
        # ==========================================
        # POST-PROCESSING & INTERPOLATION
        # ==========================================
        full_t = np.concatenate(all_t)
        full_y = np.hstack(all_y)
        
        # --- THE FIX: Filter out duplicate micro-steps to prevent interp1d divide-by-zero ---
        diffs = np.diff(full_t)
        # Keep the first element, and any element that advances time by more than a tiny margin
        keep_idx = np.insert(diffs > 1e-10, 0, True) 
        
        full_t_clean = full_t[keep_idx]
        full_y_clean = full_y[:, keep_idx]
        
        # Interpolate the clean, variable-step data back onto the user's smooth tspan
        from scipy.interpolate import interp1d
        interpolator = interp1d(full_t_clean, full_y_clean, axis=1, kind='linear', fill_value="extrapolate")
        
        self.tspan = tspan
        self.result = interpolator(tspan)
        nt = len(self.tspan)
        
        # Initialize storage for V and beta_dot
        self.V = [None] * nt
        self.beta_dot = [None] * nt
        
        # Post-processing loop: Re-evaluate to extract spatial velocities
        for i in range(nt):
            t_val = self.tspan[i]
            y_val = self.result[:, i]
            
            state_dot, V_val = ODEfun(t_val, y_val, V_base, A_base)
            
            self.V[i] = V_val
            self.beta_dot[i] = state_dot[self.total_nq:]

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Simulation finished. Runtime: {elapsed_time:.2f} s")


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