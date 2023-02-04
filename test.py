import numpy as np
import matplotlib.pyplot as plt
from quadrotor import Quad
from traj import *


npzfile = np.load('test.npz')
x = npzfile['arr_0']
u = npzfile['arr_1']
pt = x[:,:3].T

sim_dt = 0.05  #simulation sample time
mpc_dt = 0.1   # mpc sample time
quad = Quad(noisy=False, drag=False, 
                motor_noisy=False, rotor_dyn=False, quad_config='Default')
""" generate reference trajectory and controls """
traj_ref, t_ref, u_ref = loop_trajectory(quad,mpc_dt,plot=False,yawing=True)

l = t_ref.shape[0]
x_ol = np.zeros((l,3))
for i in range(l):
    for _ in range(int(mpc_dt/sim_dt)):
        quad.update(u_ref[i,:],dt=sim_dt)
    x_ol[i,:] = quad.x


# plt.plot(x[:,0])
# plt.plot(traj_ref[:,0])
# plt.plot(x_ol[:,0])

draw_poly(traj_ref, u_ref, t_ref, target_points=pt, target_t=None)


# plt.show()