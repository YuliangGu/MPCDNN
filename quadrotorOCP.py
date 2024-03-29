import numpy as np
import tensorflow as tf
from tensorflow import keras
import casadi as cs
from quadrotor import Quad
from utils import *

class QuadOpt:   
    def __init__(self, quad, residual = None, t_horizon = 1, control_nodes = 10,
                 Q_cost=None, R_cost = None,
                 drag = False):
        """
        :param quad: Quad object
        :param residual: nominal + learned residual model. A boolean value. False if nominal is used 
        :param t_horizon: time horizon for MPC
        :param control_nodes: number of control nodes until time horizon
        :param Q_cost: cost for state. A numpy array of size 13. None if use default
        :param R_cost: cost for controls. A numpy array of size 4. None if use default
        :drag: a linear drag model. A boolean value. False if not used
        """
        
        self.quad = quad
        self.T = t_horizon 
        self.N = control_nodes 
        
        if Q_cost is None:
            self.Q = np.array([20.0,20.0,20.0,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05,0.05])
        if R_cost is None:
            self.R = 0.5 * np.ones(4)
            
        # Declare model variables
        self.x = cs.MX.sym('x', 3)   # position
        self.v = cs.MX.sym('v', 3)   # velocity
        self.quat = cs.MX.sym('quat', 4)   # quat
        self.W = cs.MX.sym('W', 3)   # angle rate
        
        # Full state vector
        self.X = cs.vertcat(self.x, self.v, self.quat, self.W)
        self.state_dim = 13
        
        # Control input vector
        u1 = cs.MX.sym('u1')
        u2 = cs.MX.sym('u2')
        u3 = cs.MX.sym('u3')
        u4 = cs.MX.sym('u4')
        self.u = cs.vertcat(u1, u2, u3, u4)
        self.u_lb = [0,0,0,0]
        self.u_ub = [Quad.max_thrust,Quad.max_thrust,Quad.max_thrust,Quad.max_thrust]
        
        # Nominal model equations symbolic function (no NN)
        self.quad_xdot_nominal = self.quad_dynamics(drag)
        self.quad_xdot = self.quad_xdot_nominal

        # Cost and reference (to be overwritten)
        self.X_ref = None
        self.u_ref = None
        self.X_ref_traj = None
        self.u_ref_traj = None
        self.L = None

        self.mlp = None
        self.mlp_conf = None

        if residual is not None:
            # pass the DNN model to the class
            self.mlp = residual['model']
            self.mlp_conf = residual['conf']
            self.quad_xdot_aug = self.quad_dynamics_aug(self.quad_xdot_nominal(X=self.X, u=self.u)['X_dot'])
            self.quad_xdot = self.quad_xdot_aug

        # Declare model variables for NN prediction
        self.nn_p = cs.MX.sym('nn_p', 3)
        self.nn_v = cs.MX.sym('nn_v', 3)
        self.nn_quat = cs.MX.sym('nn_quat', 4)
        self.nn_W = cs.MX.sym('nn_W', 3)
        self.nn_X = cs.vertcat(self.nn_p, self.nn_v, self.nn_quat,self.nn_W)

    def linearized_quad_dynamics(self):
        """
        Jacobian J matrix of the linearized dynamics specified in the function quad_dynamics. J[i, j] corresponds to
        the partial derivative of f_i(x) wrt x(j).
        :return: a CasADi symbolic function that calculates the 13 x 13 Jacobian matrix of the linearized simplified
        quadrotor dynamics
        """

        jac = cs.MX(self.state_dim, self.state_dim)

        # Position derivatives
        jac[0:3, 7:10] = cs.diag(cs.MX.ones(3))

        # Angle derivatives
        jac[3:7, 3:7] = skew_symmetric(self.r) / 2
        jac[3, 10:] = 1 / 2 * cs.horzcat(-self.q[1], -self.q[2], -self.q[3])
        jac[4, 10:] = 1 / 2 * cs.horzcat(self.q[0], -self.q[3], self.q[2])
        jac[5, 10:] = 1 / 2 * cs.horzcat(self.q[3], self.q[0], -self.q[1])
        jac[6, 10:] = 1 / 2 * cs.horzcat(-self.q[2], self.q[1], self.q[0])

        # Velocity derivatives
        a_u = (self.u[0] + self.u[1] + self.u[2] + self.u[3]) * self.quad.max_thrust / self.quad.mass
        jac[7, 3:7] = 2 * cs.horzcat(a_u * self.q[2], a_u * self.q[3], a_u * self.q[0], a_u * self.q[1])
        jac[8, 3:7] = 2 * cs.horzcat(-a_u * self.q[1], -a_u * self.q[0], a_u * self.q[3], a_u * self.q[2])
        jac[9, 3:7] = 2 * cs.horzcat(0, -2 * a_u * self.q[1], -2 * a_u * self.q[1], 0)

        # Rate derivatives
        jac[10, 10:] = (self.quad.J[1] - self.quad.J[2]) / self.quad.J[0] * cs.horzcat(0, self.r[2], self.r[1])
        jac[11, 10:] = (self.quad.J[2] - self.quad.J[0]) / self.quad.J[1] * cs.horzcat(self.r[2], 0, self.r[0])
        jac[12, 10:] = (self.quad.J[0] - self.quad.J[1]) / self.quad.J[2] * cs.horzcat(self.r[1], self.r[0], 0)

        return cs.Function('J', [self.x, self.u], [jac])
        
    def discretize_f_and_q(self, m):
        """
        m: steps per control intervals
        """
        return discretize_dynamics_and_cost(self.T, self.N, m, self.X, self.u, self.quad_xdot, self.cost_f())
        
    def cost_f(self):
        """
        Symbolic cost function given reference values
        """
        x_e = self.X[:3] - self.X_ref[:3]
        v_e = self.X[3:6] - self.X_ref[3:6]
        quat_e = q_dot_q(self.X[6:10], quaternion_inverse(self.X_ref[6:10]))
        W_e = self.X[10:] - self.X_ref[10:]

        X_e = cs.vertcat(x_e,v_e,quat_e,W_e)
        u_e = self.u - self.u_ref

        state_cost = (self.Q * X_e).T @ X_e 
        control_cost = (self.R * u_e).T @ u_e 
        # not adding terminal cost yet
        q = state_cost + control_cost
        return cs.Function('q', [self.X,self.u],[q],['X','u'],['q'])
        
    def quad_dynamics_aug(self, nominal):
        """
        CasADi function of DNN augmented dynamcis
        """
        state = self.X
        
        v_b = v_dot_q(state[3:6], quaternion_inverse(state[6:10])) # convert to body frame
        state = cs.vertcat(state[:3], v_b, state[6:])
        mlp_input = cs.vertcat(state, self.u)
        mlp_output = self.mlp.forward(mlp_input)
        if self.mlp_conf['full_state']:
            out_force_b = mlp_output[3:6]
            out_force = v_dot_q(out_force_b, state[6:10])
            mlp_output = cs.vertcat(mlp_output[:3],out_force, mlp_output[3:])
        if self.mlp_conf['reduced_state']: #only learns v and W dynamics
            out_force_b = mlp_output[:3]
            out_force = v_dot_q(out_force_b, state[6:10])
            mlp_output = cs.vertcat([0,0,0],mlp_output[:3],[0,0,0,0],mlp_output[3:])
        X_dot_aug = nominal + mlp_output
        return cs.Function('X_dot', [self.X, self.u], [X_dot_aug], ['X','u'], ['X_dot'])

    def quad_dynamics(self, drag):
        """
        Symbolic dynamics of the 3D quadrotor model. 
        return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        """
        X_dot = cs.vertcat(self.x_dynamics(), self.v_dynamics(drag), self.q_dynamics(), self.W_dynamics())
        return cs.Function('X_dot', [self.X[:13], self.u], [X_dot], ['X', 'u'], ['X_dot'])
        
    def x_dynamics(self):
        return self.v
    
    def v_dynamics(self, drag):
        f_thrust = self.u
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3]) / self.quad.m
        v_dynamics = v_dot_q(a_thrust, self.quat) - g
        
        if not drag:
            # Velocity in body frame:
            v_b = v_dot_q(self.v, quaternion_inverse(self.quat))
            a_drag = cs.vertcat(-self.quad.kdx * v_b[0],
                                -self.quad.kdy * v_b[1],
                                -self.quad.kdz * v_b[2] + self.quad.kdh * (v_b[0]**2 + v_b[1]**2)) / self.quad.m
            v_dynamics += a_drag
            
        return v_dynamics
    
    def q_dynamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.W), self.quat)
    
    def W_dynamics(self):
        f_thrust = self.u
        G2 = cs.MX(self.quad.G2)
        G3 = cs.MX(self.quad.G3)
        G4 = cs.MX(self.quad.G4)
        return cs.vertcat(
            (cs.mtimes(f_thrust.T, G2) + (self.quad.J[1] - self.quad.J[2]) * self.W[1] * self.W[2]) / self.quad.J[0],
            (cs.mtimes(f_thrust.T, G3) + (self.quad.J[2] - self.quad.J[0]) * self.W[2] * self.W[0]) / self.quad.J[1],
            (cs.mtimes(f_thrust.T, G4) + (self.quad.J[0] - self.quad.J[1]) * self.W[0] * self.W[1]) / self.quad.J[2])
    
    def set_reference_state(self, X_ref=None, u_ref=None):
        if X_ref is None:
            X_ref = [0,0,0,0,0,0,1,0,0,0,0,0,0]
        if u_ref is None:
            u_ref = [0, 0, 0, 0]   
        # convert the ref velocity to the bodyframe
        v_b = v_dot_q(X_ref[3:6], quaternion_inverse(X_ref[6:10]))
        X_ref = np.concatenate((X_ref[:3], v_b, X_ref[6:]))
        self.X_ref = X_ref
        self.u_ref = u_ref
    
    def set_reference_traj(self, X_ref_traj=None, u_ref_traj=None):
        """set reference traj"""
        # If not enough states in target sequence, append last state until required length is met
        while X_ref_traj.shape[0] < self.N + 1:
            X_ref_traj = np.concatenate((X_ref_traj, np.expand_dims(X_ref_traj[-1, :], 0)), 0)
            if u_ref_traj is not None:
                u_ref_traj = np.concatenate((u_ref_traj, np.expand_dims(u_ref_traj[-1, :], 0)), 0)         
        # stacked_X_ref_traj = np.concatenate([x for x in X_ref_traj], 1)
        # Tranform to velocity to body frame?
        self.X_ref_traj = X_ref_traj
        self.u_ref_traj = u_ref_traj
        
    
    def OCP(self, X_init = None):
        """
        :params X_init: initial conditons from quadrotor. to be upated after each OCP
        """
        # direct multiple shooting 
        if X_init is None:
            X_init = np.array([0,0,0,0,0,0,1,0,0,0,0,0,0])
        
        # starting with an empty NLP
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0.0
        g = []
        lbg = []
        ubg = []
        
        # Lift initial conditions      
        Xk = cs.MX.sym('X0',13)
        w += [Xk]
        lbw += list(X_init)
        ubw += list(X_init)
        w0 += list(X_init)
        
        # Formulate NLP
        m = 4 # RK4 steps per integration
        for k in range(self.N):
            # New NLP variable for the control
            Uk = cs.MX.sym('U_'+ str(k), 4)
            w += [Uk]
            lbw += self.u_lb
            ubw += self.u_ub
            w0 += self.u_ub  # a guess for the primal-dual
        
            # set reference and integrate till the end of the interval
            self.set_reference_state(self.X_ref_traj[k,:],self.u_ref_traj[k,:])
            F = self.discretize_f_and_q(m)
            Fk = F(X0=Xk,p=Uk)
            Xk_end = Fk['Xf']
            J = J + Fk['qf']
            
            # New NLP variable for state at end of interval
            Xk = cs.MX.sym('X_' + str(k+1), 13)
            w += [Xk]
            lbw += [-3] * 13
            ubw += [3] * 13
            w0  += [0,0,0,0,0,0,0,0,0,0,0,0,0]
            
            # Add equality constraint
            g += [Xk_end-Xk]
            lbg += [0,0,0,0,0,0,0,0,0,0,0,0,0]
            ubg += [0,0,0,0,0,0,0,0,0,0,0,0,0]
        
        # Create an NLP solver
        mysolver = 'sqpmethod'
        # mysolver = 'ipopt'
        opts = {}
        opts['qpsol_options'] = {'error_on_fail': False}

        prob = {'f': J, 'x': cs.vertcat(*w), 'g': cs.vertcat(*g)}
        solver = cs.nlpsol('solver', mysolver, prob, opts)
        sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
        
        return sol['x'].full().flatten()