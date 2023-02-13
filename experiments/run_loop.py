from quadrotor import Quad
from quadrotorOCP import QuadOpt
from quadrotorMPC import QuadMPC
from traj import *

sim_dt = 0.005  #simulation sample time
mpc_dt = 0.01   # mpc sample time
quad = Quad(noisy=False, drag=False, 
                motor_noisy=False, rotor_dyn=False, quad_config='Default')

""" generate reference trajectory and controls """
traj_ref, t_ref, u_ref = loop_trajectory(quad,mpc_dt,plot=False,yawing=False)
x_ref = traj_ref[:,:3]
q_ref = traj_ref[:,3:7]
v_ref = traj_ref[:,7:10]
W_ref = traj_ref[:,10:]
traj_ref = np.concatenate((x_ref,v_ref,q_ref,W_ref),1)

""" set initil condition """
X_init = traj_ref[0,:]
quad.set_state(X_init)

""" initilize the MPC """
N_nodes = 10
t_horizon = N_nodes * mpc_dt
quadmpc = QuadMPC(quad, t_horizon=t_horizon, N_nodes=N_nodes,
                Q_cost=None, R_cost=None, opt_dt = mpc_dt, sim_dt=sim_dt,
                 NN_model=False, drag_model=False)

""" start the simulation and save the results"""
quadmpc.set_ref(traj_ref=traj_ref, u_ref=u_ref)     
x, u = quadmpc.simulate()
np.savez('data/test.npz', x, u)

