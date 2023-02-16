import numpy as np
import matplotlib.pyplot as plt
from quadrotor import Quad
from traj import *
from scipy import signal
from sklearn.metrics import mean_squared_error
from mpl_toolkits import mplot3d

"""reference trajectory and controls """
sim_dt = 0.01 #simulation sample time
mpc_dt = 0.1   # mpc sample time
quad = Quad()
traj_ref, t_ref, u_ref = loop_trajectory(quad,mpc_dt,plot=False,yawing=False)

""" load data """
npzfile = np.load('data_2learn/test_mlp_2.npz')
x_mpc_dnn = npzfile['arr_0'][:,:3]
u_opt_dnn = npzfile['arr_1']

""" load data  """
npzfile = np.load('data_2learn/test_1.npz')
x_mpc = npzfile['arr_0'][:,:3]
u_mpc = npzfile['arr_1']

"""plot results"""
# plt.plot(x_mpc[:,0],x_mpc[:,1],label='mpc', color='g')
# plt.plot(traj_ref[:,0],traj_ref[:,1],label='ref',color='r')
# plt.plot(x_mpc_dnn[:,0],x_mpc_dnn[:,1], label='mpc_dnn', color='b')

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(x_mpc[:,0],x_mpc[:,1],x_mpc[:,2],label='mpc')
ax.plot3D(traj_ref[:,0],traj_ref[:,1],traj_ref[:,2],label='ref')
ax.plot3D(x_mpc_dnn[:,0],x_mpc_dnn[:,1],x_mpc_dnn[:,2],label ='mpc_dnn')

""" plot the power spectum"""
fs = 20
f, Pxx_den = signal.periodogram(x_opt[:,1],fs)
f2, Pxx_den2 = signal.periodogram(traj_ref[:,1],fs)
plt.semilogy(f, Pxx_den,label='rotor')
plt.semilogy(f2, Pxx_den2)

plt.legend()
plt.show()

