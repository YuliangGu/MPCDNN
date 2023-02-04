import numpy as np
import matplotlib.pyplot as plt
from utils import *

def loop_trajectory(quad, discretization_dt, radius=1, z=1, lin_acc=1.0, clockwise=True, yawing=False, v_max=2.5, plot=False):
    """
    Creates a circular trajectory on the x-y plane that increases speed by 1m/s at every revolution.
    :param quad: Quadrotor model
    :param discretization_dt: Sampling period of the trajectory.|
    :param radius: radius of loop trajectory in meters
    :param z: z position of loop plane in meters
    :param lin_acc: linear acceleration of trajectory (and successive deceleration) in m/s^2
    :param clockwise: True if the rotation will be done clockwise.
    :param yawing: True if the quadrotor yaws along the trajectory. False for 0 yaw trajectory.
    :param v_max: Maximum speed at peak velocity. Revolutions needed will be calculated automatically.
    :param map_name: Name of map to load its limits
    :param plot: Whether to plot an analysis of the planned trajectory or not.
    :return: The full 13-DoF trajectory with time and input vectors
    """
    assert z > 0

    ramp_up_t = 2  # s

    # Calculate simulation time to achieve desired maximum velocity with specified acceleration
    t_total = 2 * v_max / lin_acc + 2 * ramp_up_t

    # Transform to angular acceleration
    alpha_acc = lin_acc / radius  # rad/s^2

    # Generate time and angular acceleration sequences
    # Ramp up sequence
    ramp_t_vec = np.arange(0, ramp_up_t, discretization_dt)
    ramp_up_alpha = alpha_acc * np.sin(np.pi / (2 * ramp_up_t) * ramp_t_vec) ** 2
    # Acceleration phase
    coasting_duration = (t_total - 4 * ramp_up_t) / 2
    coasting_t_vec = ramp_up_t + np.arange(0, coasting_duration, discretization_dt)
    coasting_alpha = np.ones_like(coasting_t_vec) * alpha_acc
    # Transition phase: decelerate
    transition_t_vec = np.arange(0, 2 * ramp_up_t, discretization_dt)
    transition_alpha = alpha_acc * np.cos(np.pi / (2 * ramp_up_t) * transition_t_vec)
    transition_t_vec += coasting_t_vec[-1] + discretization_dt
    # Deceleration phase
    down_coasting_t_vec = transition_t_vec[-1] + np.arange(0, coasting_duration, discretization_dt) + discretization_dt
    down_coasting_alpha = -np.ones_like(down_coasting_t_vec) * alpha_acc
    # Bring to rest phase
    ramp_up_t_vec = down_coasting_t_vec[-1] + np.arange(0, ramp_up_t, discretization_dt) + discretization_dt
    ramp_up_alpha_end = ramp_up_alpha - alpha_acc

    # Concatenate all sequences
    t_ref = np.concatenate((ramp_t_vec, coasting_t_vec, transition_t_vec, down_coasting_t_vec, ramp_up_t_vec))
    alpha_vec = np.concatenate((
        ramp_up_alpha, coasting_alpha, transition_alpha, down_coasting_alpha, ramp_up_alpha_end))

    # Calculate derivative of angular acceleration (alpha_vec)
    ramp_up_alpha_dt = alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / ramp_up_t * ramp_t_vec)
    coasting_alpha_dt = np.zeros_like(coasting_alpha)
    transition_alpha_dt = - alpha_acc * np.pi / (2 * ramp_up_t) * np.sin(np.pi / (2 * ramp_up_t) * transition_t_vec)
    alpha_dt = np.concatenate((
        ramp_up_alpha_dt, coasting_alpha_dt, transition_alpha_dt, coasting_alpha_dt, ramp_up_alpha_dt))

    if not clockwise:
        alpha_vec *= -1
        alpha_dt *= -1

    # Compute angular integrals
    w_vec = np.cumsum(alpha_vec) * discretization_dt
    angle_vec = np.cumsum(w_vec) * discretization_dt

    # Compute position, velocity, acceleration, jerk
    pos_traj_x = radius * np.sin(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_y = radius * np.cos(angle_vec)[np.newaxis, np.newaxis, :]
    pos_traj_z = np.ones_like(pos_traj_x) * z

    vel_traj_x = (radius * w_vec * np.cos(angle_vec))[np.newaxis, np.newaxis, :]
    vel_traj_y = - (radius * w_vec * np.sin(angle_vec))[np.newaxis, np.newaxis, :]

    acc_traj_x = radius * (alpha_vec * np.cos(angle_vec) - w_vec ** 2 * np.sin(angle_vec))[np.newaxis, np.newaxis, :]
    acc_traj_y = - radius * (alpha_vec * np.sin(angle_vec) + w_vec ** 2 * np.cos(angle_vec))[np.newaxis, np.newaxis, :]

    jerk_traj_x = radius * (alpha_dt * np.cos(angle_vec) - alpha_vec * np.sin(angle_vec) * w_vec -
                            np.cos(angle_vec) * w_vec ** 3 - 2 * np.sin(angle_vec) * w_vec * alpha_vec)
    jerk_traj_y = - radius * (np.cos(angle_vec) * w_vec * alpha_vec + np.sin(angle_vec) * alpha_dt -
                              np.sin(angle_vec) * w_vec ** 3 + 2 * np.cos(angle_vec) * w_vec * alpha_vec)
    jerk_traj_x = jerk_traj_x[np.newaxis, np.newaxis, :]
    jerk_traj_y = jerk_traj_y[np.newaxis, np.newaxis, :]

    if yawing:
        yaw_traj = -angle_vec
    else:
        yaw_traj = np.zeros_like(angle_vec)

    traj = np.concatenate((
        np.concatenate((pos_traj_x, pos_traj_y, pos_traj_z), 1),
        np.concatenate((vel_traj_x, vel_traj_y, np.zeros_like(vel_traj_x)), 1),
        np.concatenate((acc_traj_x, acc_traj_y, np.zeros_like(acc_traj_x)), 1),
        np.concatenate((jerk_traj_x, jerk_traj_y, np.zeros_like(jerk_traj_x)), 1)), 0)

    yaw = np.concatenate((yaw_traj[np.newaxis, :], w_vec[np.newaxis, :]), 0)

    return minimum_snap_trajectory_generator(traj, yaw, t_ref, quad, plot)


def minimum_snap_trajectory_generator(traj_derivatives, yaw_derivatives, t_ref, quad, plot):
    """
    Follows the Minimum Snap Trajectory paper to generate a full trajectory given the position reference and its
    derivatives, and the yaw trajectory and its derivatives.
    
    :param traj_derivatives: np.array of shape 4x3xN. N corresponds to the length in samples of the trajectory, and:
        - The 4 components of the first dimension correspond to position, velocity, acceleration and jerk.
        - The 3 components of the second dimension correspond to x, y, z.
        
    :param yaw_derivatives: np.array of shape 2xN. N corresponds to the length in samples of the trajectory. The first
    row is the yaw trajectory, and the second row is the yaw time-derivative trajectory.
    
    :param t_ref: vector of length N, containing the reference times (starting from 0) for the trajectory.
    :param quad: Quadrotor3D object, corresponding to the quadrotor model that will track the generated reference.
    :type quad: Quadrotor3D
    :param plot: True if show a plot of the generated trajectory.
    :return: tuple of 3 arrays:
        - Nx13 array of generated reference trajectory. The 13 dimension contains the components: position_xyz,
        attitude_quaternion_wxyz, velocity_xyz, body_rate_xyz.
        - N array of reference timestamps. The same as in the input
        - Nx4 array of reference controls, corresponding to the four motors of the quadrotor.
    """

    discretization_dt = t_ref[1] - t_ref[0]
    len_traj = traj_derivatives.shape[2]

    # Add gravity to accelerations
    gravity = 9.81
    thrust = traj_derivatives[2, :, :].T + np.tile(np.array([[0, 0, 1]]), (len_traj, 1)) * gravity
    # Compute body axes
    z_b = thrust / np.sqrt(np.sum(thrust ** 2, 1))[:, np.newaxis]

    yawing = np.any(yaw_derivatives[0, :] != 0)

    rate = np.zeros((len_traj, 3))
    f_t = np.zeros((len_traj, 1))
    for i in range(len_traj):
        f_t[i, 0] = quad.m * z_b[i].dot(thrust[i, :].T)

    if yawing:
        # yaw is defined as the projection of the body-x axis on the horizontal plane
        x_c = np.concatenate((np.cos(yaw_derivatives[0, :])[:, np.newaxis],
                              np.sin(yaw_derivatives[0, :])[:, np.newaxis],
                              np.zeros(len_traj)[:, np.newaxis]), 1)
        y_b = np.cross(z_b, x_c)
        y_b = y_b / np.sqrt(np.sum(y_b ** 2, axis=1))[:, np.newaxis]
        x_b = np.cross(y_b, z_b)

        # Rotation matrix (from body to world)
        b_r_w = np.concatenate((x_b[:, :, np.newaxis], y_b[:, :, np.newaxis], z_b[:, :, np.newaxis]), -1)
        q = []
        for i in range(len_traj):
            # Transform to quaternion
            q.append(rotation_matrix_to_quat(b_r_w[i]))
            if i > 1:
                q[-1] = undo_quaternion_flip(q[-2], q[-1])
        q = np.stack(q)

        # Compute angular rate vector
        # Total thrust acceleration must be equal to the projection of the quadrotor acceleration into the Z body axis
        a_proj = np.zeros((len_traj, 1))

        for i in range(len_traj):
            a_proj[i, 0] = z_b[i].dot(traj_derivatives[3, :, i])

        h_omega = quad.m / f_t * (traj_derivatives[3, :, :].T - a_proj * z_b)
        for i in range(len_traj):
            rate[i, 0] = -h_omega[i].dot(y_b[i])
            rate[i, 1] = h_omega[i].dot(x_b[i])
            rate[i, 2] = -yaw_derivatives[1, i] * np.array([0, 0, 1]).dot(z_b[i])

    else:
        # new way to compute attitude:
        # https://math.stackexchange.com/questions/2251214/calculate-quaternions-from-two-directional-vectors
        e_z = np.array([[0.0, 0.0, 1.0]])
        q_w = 1.0 + np.sum(e_z * z_b, axis=1)
        q_xyz = np.cross(e_z, z_b)
        q = 0.5 * np.concatenate([np.expand_dims(q_w, axis=1), q_xyz], axis=1)
        q = q / np.sqrt(np.sum(q ** 2, 1))[:, np.newaxis]

        # Use numerical differentiation of quaternions
        q_dot = np.gradient(q, axis=0) / discretization_dt
        w_int = np.zeros((len_traj, 3))
        for i in range(len_traj):
            w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]
        rate[:, 0] = w_int[:, 0]
        rate[:, 1] = w_int[:, 1]
        rate[:, 2] = w_int[:, 2]

        go_crazy_about_yaw = True
        if go_crazy_about_yaw:
            print("Maximum yawrate before adaption: %.3f" % np.max(np.abs(rate[:, 2])))
            q_new = q
            yaw_corr_acc = 0.0
            for i in range(1, len_traj):
                yaw_corr = -rate[i, 2] * discretization_dt
                yaw_corr_acc += yaw_corr
                q_corr = np.array([np.cos(yaw_corr_acc / 2.0), 0.0, 0.0, np.sin(yaw_corr_acc / 2.0)])
                q_new[i, :] = q_dot_q(q[i, :], q_corr)
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q[i, :]), q_dot[i])[1:]

            q_new_dot = np.gradient(q_new, axis=0) / discretization_dt
            for i in range(1, len_traj):
                w_int[i, :] = 2.0 * q_dot_q(quaternion_inverse(q_new[i, :]), q_new_dot[i])[1:]

            q = q_new
            rate[:, 0] = w_int[:, 0]
            rate[:, 1] = w_int[:, 1]
            rate[:, 2] = w_int[:, 2]
            print("Maximum yawrate after adaption: %.3f" % np.max(np.abs(rate[:, 2])))

    # Compute inputs
    rate_dot = np.gradient(rate, axis=0) / discretization_dt

    rate_x_Jrate = np.array([(quad.J[2] - quad.J[1]) * rate[:, 2] * rate[:, 1],
                             (quad.J[0] - quad.J[2]) * rate[:, 0] * rate[:, 2],
                             (quad.J[1] - quad.J[0]) * rate[:, 1] * rate[:, 0]]).T

    tau = rate_dot * quad.J[np.newaxis, :] + rate_x_Jrate
    
    b = np.concatenate((tau, f_t), axis=-1)
      
    a_mat = np.concatenate((quad.G2[np.newaxis, :], quad.G3[np.newaxis, :],
                            quad.G4[np.newaxis, :], np.ones_like(quad.G4)[np.newaxis, :]), 0)

    reference_u = np.zeros((len_traj, 4))
    for i in range(len_traj):
        reference_u[i, :] = np.linalg.solve(a_mat, b[i, :])

    full_pos = traj_derivatives[0, :, :].T
    full_vel = traj_derivatives[1, :, :].T
    reference_traj = np.concatenate((full_pos, q, full_vel, rate), 1)

    reference_traj[:, 0] -= reference_traj[0, 0]
    reference_traj[:, 1] -= reference_traj[0, 1]

    if plot:
        draw_poly(reference_traj, reference_u, t_ref)

    # Change format of reference input to motor activation, in interval [0, 1]
#     reference_u = reference_u / quad.max_thrust

    return reference_traj, t_ref, reference_u

def draw_poly(traj, u_traj, t, target_points=None, target_t=None):
    """
    Plots the generated trajectory of length n with the used keypoints.
    :param traj: Full generated reference trajectory. Numpy array of shape nx13
    :param u_traj: Generated reference inputs. Numpy array of shape nx4
    :param t: Timestamps of the references. Numpy array of length n
    :param target_points: m position keypoints used for trajectory generation. Numpy array of shape 3 x m.
    :param target_t: Timestamps of the reference position keypoints. If not passed, then they are extracted from the
    t vector, assuming constant time separation.
    """

    ders = 2
    dims = 3

    y_labels = [r'pos $[m]$', r'vel $[m/s]$', r'acc $[m/s^2]$', r'jer $[m/s^3]$']
    dim_legends = ['x', 'y', 'z']

    if target_t is None and target_points is not None:
        target_t = np.linspace(0, t[-1], target_points.shape[1])

    p_traj = traj[:, :3]
    a_traj = traj[:, 3:7]
    v_traj = traj[:, 7:10]
    r_traj = traj[:, 10:]

    plt_traj = [p_traj, v_traj]

    fig = plt.figure()
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1)

        for dim in range(dims):

            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[dim])

            if d_ord == 0 and target_points is not None:
                plt.plot(target_t, target_points[dim, :], 'bo')

        plt.gca().set_xticklabels([])
        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])

    dim_legends = [['w', 'x', 'y', 'z'], ['x', 'y', 'z']]
    y_labels = [r'att $[quat]$', r'rate $[rad/s]$']
    plt_traj = [a_traj, r_traj]
    for d_ord in range(ders):

        plt.subplot(ders + 2, 2, d_ord * 2 + 1 + ders * 2)
        for dim in range(plt_traj[d_ord].shape[1]):
            plt.plot(t, plt_traj[d_ord][:, dim], label=dim_legends[d_ord][dim])

        plt.legend()
        plt.grid()
        plt.ylabel(y_labels[d_ord])
        if d_ord == ders - 1:
            plt.xlabel(r'time $[s]$')
        else:
            plt.gca().set_xticklabels([])

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    plt.plot(p_traj[:, 0], p_traj[:, 1], p_traj[:, 2])
    if target_points is not None:
        plt.plot(target_points[0, :], target_points[1, :], target_points[2, :], 'bo')
    plt.title('Target position trajectory')
    ax.set_xlabel(r'$p_x [m]$')
    ax.set_ylabel(r'$p_y [m]$')
    ax.set_zlabel(r'$p_z [m]$')

    plt.subplot(ders + 1, 2, (ders + 1) * 2)
    for i in range(u_traj.shape[1]):
        plt.plot(t, u_traj[:, i], label=r'$u_{}$'.format(i))
    plt.grid()
    plt.legend()
    plt.gca().yaxis.set_label_position("right")
    plt.gca().yaxis.tick_right()
    plt.xlabel(r'time $[s]$')
    plt.ylabel(r'single thrusts $[N]$')
    plt.title('Control inputs')

    plt.suptitle('Generated polynomial trajectory')

    plt.show()