import numpy as np
import matplotlib.pyplot as plt
from quadrotor import Quad
from traj import *


""" load data"""
npzfile = np.load('data/test.npz')
X_opt = npzfile['arr_0']
x_opt = X_opt[:,:3]
u_opt = npzfile['arr_1']

"""reference trajectory and controls """
sim_dt = 0.01 #simulation sample time
mpc_dt = 0.1   # mpc sample time
quad = Quad(noisy=True, drag=False, 
                motor_noisy=False, rotor_dyn=False, quad_config='Default')
traj_ref, t_ref, u_ref = loop_trajectory(quad,mpc_dt,plot=False,yawing=False)

"""open loop"""
l = t_ref.shape[0]
x_ol = np.zeros((l,3))
for i in range(l):
    for _ in range(int(mpc_dt/sim_dt)):
        quad.update(u_ref[i,:],dt=sim_dt)
    x_ol[i,:] = quad.x

"""plot results"""
plt.plot(x_opt, label='mpc', color='r')
# plt.plot(x_ol,label='open loop', color='b')
plt.plot(traj_ref[:,:3],label='ref',color='g')
plt.legend()
plt.show()
