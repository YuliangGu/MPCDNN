import numpy as np
from quadrotor import Quad
from quadrotorOCP import QuadOpt

class QuadMPC:
    def __init__(self, quadrotor, t_horizon=1.0, N_nodes=10,
                Q_cost=None, R_cost=None, opt_dt=1e-2, sim_dt=1e-3,
                NN_model=False, drag_model=False):
        """
        :param quadrotor: Quad object
        :param t_horizon: time horizon of MPC
        :param N_nodes: number of control nodes of MPC
        :param Q_cost: state cost matrix. A numpy array of siee (13,)
        :param R_cost: control cost matrix. A numpy array of size (4,)
        :param opt_dt: optimization time step
        :param sim_dt: simulation time step
        :param NN_model: None if not DNN residual model
        :param drag_model: False if not use the linear drag model
        """

        self.quad = quadrotor
        self.T = t_horizon
        self.N = N_nodes
        self.len_traj = None

        self.opt_dt = opt_dt
        self.sim_dt = sim_dt
        
        self.X_ref_list = []
        self.u_ref_list = []

        self.quad_opt = QuadOpt(quadrotor, residual=NN_model, t_horizon=t_horizon,
                                control_nodes=N_nodes, Q_cost=Q_cost, R_cost=R_cost,
                                drag=drag_model)
    
    def get_state(self,stack=True):
        return np.expand_dims(self.quad.get_state(stack=stack),1)
    
    def clear_ref(self): 
        self.X_ref_list = []
        self.u_ref_list = []

    def set_ref(self, traj_ref, u_ref):
        """
        traj_ref: reference trajectory. numpy array of size Nx13
        u_ref: reference controls. numpy array of size Nx4
        """
        self.len_traj = traj_ref.shape[0]

        l = traj_ref.shape[0]  #length of the trajectory
        for i in range(l):
            if l-i < self.N:
                X = traj_ref[i:,:]
                u = u_ref[i:,:]
            else:
                X = traj_ref[i:i+self.N,:]
                u = u_ref[i:i+self.N,:]
            self.X_ref_list.append(X)
            self.u_ref_list.append(u)

    def simulate(self):
        X_opt = np.zeros((self.len_traj, 13))
        U_opt = np.zeros((self.len_traj, 4))
        for i in range(self.len_traj):
            x_now = self.get_state()
            x_horizon = self.X_ref_list[i]
            u_horizon = self.u_ref_list[i]
            self.quad_opt.set_reference_traj(x_horizon,u_horizon) # set reference
            w_opt = self.quad_opt.OCP(X_init=x_now) # optimize
            u_opt = w_opt[13:17]
            for _ in range(int(self.opt_dt/self.sim_dt)):
                self.quad.update(u_opt, self.sim_dt) # execuate the first u
            X_opt[i,:] = np.squeeze(x_now)
            U_opt[i,:] = np.squeeze(u_opt)
        return X_opt,U_opt
    



