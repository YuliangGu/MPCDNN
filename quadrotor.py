from math import sqrt
import numpy as np
from utils import *

class Quad:
    g = np.array([[0], [0], [9.81]])
    m = 0.75
    J = np.array([2.4e-3, 2.1e-3, 4.3e-3])
    l = 0.14
    beta = 56/180*np.pi
    cq = 2.37e-8
    ct = 1.51e-6
    tau = 0.019
    
    # drag cofficients
    kdx = 0.26
    kdy = 0.28
    kdz = 0.42
    kdh = 0.01
    
    # does not consider inputs constraints yet, add it later
    max_thrust = 10.0
    min_thrust = 0.0
    
    def __init__(self, noisy=False, drag=False, 
                motor_noisy=False, rotor_dyn=False, quad_config='Default'):
        # load motor configuration
        self.G2,self.G3,self.G4 = self.motor_mixing(quad_config) # right now using the configuration from paper
            
        # System state space
        self.x = np.array([0,0,1])
        self.v = np.zeros((3,))
        self.q = np.array([1., 0., 0., 0.])  #Quaternion format: qw, qx, qy, qz
        self.W = np.zeros((3,))
        
        self.rotor_v = None
        # consider the rotor dynamics
        if rotor_dyn:
            self.rotor_v = np.zeros((4,))
            
        # controls: individual thrusts
        self.u = np.zeros((4,))
        
        # System state space with NN
        self.v_res = None
        self.W_res = None
        
        self.noisy = noisy
        self.drag = drag
        # doesn't consider motor noisy yet
        self.motor_noisy = motor_noisy
    
    def motor_mixing(self,quad_config):
        if quad_config == 'Default': # paper configuration
            h1 = Quad.l * np.sin(Quad.beta)
            h2 = Quad.l * np.cos(Quad.beta)
            h3 = Quad.cq / Quad.ct
            G2 = np.array([-h1,-h1,h1,h1])
            G3 = np.array([-h2,h2,h2,-h2])
            G4 = np.array([-h3,h3,-h3,h3])
            
        if quad_config == '1': # our configuration
            h1 = Quad.l * np.sin(Quad.beta)
            h2 = Quad.l * np.cos(Quad.beta)
            h3 = Quad.cq / Quad.ct
            G2 = np.array([-h1,h1,h1,-h1])
            G3 = np.array([h2,-h2,h2,-h2])
            G4 = np.array([h3,h3,-h3,-h3])
        return G2,G3,G4
    
    def set_state(self, X_toSet):  #Input X_toSet = [x,v,q,W]
        self.x = X_toSet[:3]
        self.v = X_toSet[3:6]
        self.q = X_toSet[6:10]
        self.W = X_toSet[10:]
        
    def get_state(self, stack=False):
        if stack:
            return [self.x[0],self.x[1],self.x[2],
                   self.v[0],self.v[1],self.v[2],
                   self.q[0],self.q[1],self.q[2],self.q[3],
                   self.W[0],self.W[1],self.W[2]]
        else:
            return [self.x,self.v,self.q,self.W]
        
    def get_control(self, noisy = False):
        return self.u
        
    def update(self, u, dt): # u : individual thrust (4,)
        
        if self.rotor_v is not None:
            rotor_v_ref = np.sqrt(u/Quad.ct)
            rotor_v_dot = 1/Quad.tau * (rotor_v_ref - self.rotor_v)
            self.u = Quad.ct * self.rotor_v**2
            
            # update rotor_v using euler int (rk4 later)
            rotor_v_next = self.rotor_v + rotor_v_dot * dt
            self.rotor_v = rotor_v_next
        else:
            self.u = u
        
        if self.noisy: # add disterbances to force and torque
            t_d = np.random.normal(size=(3, 1), scale=10 * dt)
            f_d = np.random.normal(size=(3, 1), scale=10 * dt)
        else:
            t_d = np.zeros((3,1))
            f_d = np.zeros((3,1))
            
        X = self.get_state()   
        
        # RK4 integration
        k1 = [self.f_x(X), self.f_v(X, self.u, f_d), self.f_q(X), self.f_W(X, self.u, t_d)]  
        X_ = [X[i] + dt / 2 * k1[i] for i in range(4)]
        k2 = [self.f_x(X_), self.f_v(X_, self.u, f_d), self.f_q(X_), self.f_W(X_, self.u, t_d)]
        X_ = [X[i] + dt / 2 * k2[i] for i in range(4)]      
        k3 = [self.f_x(X_), self.f_v(X_, self.u, f_d), self.f_q(X_), self.f_W(X_, self.u, t_d)]
        X_ = [X[i] + dt / 2 * k3[i] for i in range(4)]
        k4 = [self.f_x(X_), self.f_v(X_, self.u, f_d), self.f_q(X_), self.f_W(X_, self.u, t_d)]
        X = [X[i] + dt * (1.0 / 6.0 * k1[i] + 2.0 / 6.0 * k2[i] + 2.0 / 6.0 * k3[i] + 1.0 / 6.0 * k4[i]) for i in
             range(4)]
        
        X[2] = unit_quat(X[2])
        self.x, self.v, self.q, self.W = X
          
    def f_x(self,X): # X = [x,v,q,W]
        return X[1]
    
    def f_v(self,X,u,f_d): # u: individual thrust
        a_thrust = np.array([[0], [0], [np.sum(u)]]) / Quad.m
        if self.drag:
            # convert to body frame
            v_b = v_dot_q(X[1],quaternion_inverse(X[2]))[:,np.newaxis]
            a_drag = np.array([-Quad.kdx * v_b[0], 
                               -Quad.kdy * v_b[1],
                               -Quad.kdz * v_b[2] + Quad.kdh * (v_b[0]**2 + v_b[1]**2)]) / Quad.m
        else:
            a_drag = np.zeros((3,1))
        q = X[2]
        return np.squeeze(-Quad.g + a_drag + v_dot_q(a_thrust + f_d/Quad.m, q))
    
    def f_q(self, X):
        W = X[3]
        q = X[2]
        return 1 / 2 * skew_symmetric(W).dot(q)
    
    def f_W(self, X, u, t_d): # dtau = disturbance torque vector
        W = X[3]
        return np.array([
            1 / Quad.J[0] * (u.dot(self.G2) + t_d[0] + (Quad.J[1] - Quad.J[2]) * W[1] * W[2]),
            1 / Quad.J[1] * (u.dot(self.G3) + t_d[1] + (Quad.J[2] - Quad.J[0]) * W[2] * W[0]),
            1 / Quad.J[2] * (u.dot(self.G4) + t_d[2] + (Quad.J[0] - Quad.J[1]) * W[0] * W[1])
        ]).squeeze()