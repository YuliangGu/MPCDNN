from quadrotor import Quad
from quadrotorOCP import QuadOpt
from quadrotorMPC import QuadMPC
from NNmodel import MLP
from traj import *

sim_dt = 0.01  #simulation sample time
mpc_dt = 0.1   # mpc sample time
quad = Quad(noisy=True, drag=True, 
                motor_noisy=False, rotor_dyn=True, quad_config='Default')

""" generate reference trajectory and controls """
traj_ref, t_ref, u_ref = loop_trajectory(quad,mpc_dt,plot=False,yawing=False)
x_ref = traj_ref[:,:3]
q_ref = traj_ref[:,3:7]
v_ref = traj_ref[:,7:10]
W_ref = traj_ref[:,10:]
traj_ref = np.concatenate((x_ref,v_ref,q_ref,W_ref),1)
print(traj_ref.shape)

""" set initil condition """
X_init = traj_ref[0,:]
quad.set_state(X_init)

"""try NN model"""
mlp = MLP('model_mlp')
dnn = {}
dnn['model'] = mlp
dnn['conf'] = {'reduced_state':False, 'full_state':False, 'minimal_state':True}

""" initilize the MPC """
N_nodes = 4
t_horizon = N_nodes * mpc_dt
quadmpc = QuadMPC(quad, t_horizon=t_horizon, N_nodes=N_nodes,
                Q_cost=None, R_cost=None, opt_dt = mpc_dt, sim_dt=sim_dt,
                 NN_model=dnn, drag_model=False)

""" start the simulation and save the results"""
quadmpc.set_ref(traj_ref=traj_ref, u_ref=u_ref)     
x, u = quadmpc.simulate()
np.savez('data_2learn/test_mlp_2.npz', x, u)

