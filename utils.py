import numpy as np
import casadi as cs
import pyquaternion

# utils functios
def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)

    return cs.mtimes(rot_mat, v)

def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array([
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
            [2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)],
            [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)]])

    else:
        rot_mat = cs.vertcat(
            cs.horzcat(1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)),
            cs.horzcat(2 * (qx * qy + qw * qz), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qw * qx)),
            cs.horzcat(2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx ** 2 + qy ** 2)))

    return rot_mat

def unit_quat(q):

    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return: the unit quaternion in the same data format as the original one
    """

    if isinstance(q, np.ndarray):
        q_norm = np.sqrt(np.sum(q ** 2))
    else:
        q_norm = cs.sqrt(cs.sumsqr(q))
    return 1 / q_norm * q

def quaternion_inverse(q):
    w, x, y, z = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        return np.array([w, -x, -y, -z])
    else:
        return cs.vertcat(w, -x, -y, -z)

def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)
    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """
    if isinstance(v, np.ndarray):
        return np.array([[0, -v[0], -v[1], -v[2]],
                         [v[0], 0, v[2], -v[1]],
                         [v[1], -v[2], 0, v[0]],
                         [v[2], v[1], -v[0], 0]])

    return cs.vertcat(
        cs.horzcat(0, -v[0], -v[1], -v[2]),
        cs.horzcat(v[0], 0, v[2], -v[1]),
        cs.horzcat(v[1], -v[2], 0, v[0]),
        cs.horzcat(v[2], v[1], -v[0], 0))

def q_dot_q(q, r):
    """
    Applies the rotation of quaternion r to quaternion q. In order words, rotates quaternion q by r. Quaternion format:
    wxyz.
    :param q: 4-length numpy array or CasADi MX. Initial rotation
    :param r: 4-length numpy array or CasADi MX. Applied rotation
    :return: The quaternion q rotated by r, with the same format as in the input.
    """

    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
    rw, rx, ry, rz = r[0], r[1], r[2], r[3]

    t0 = rw * qw - rx * qx - ry * qy - rz * qz
    t1 = rw * qx + rx * qw - ry * qz + rz * qy
    t2 = rw * qy + rx * qz + ry * qw - rz * qx
    t3 = rw * qz - rx * qy + ry * qx + rz * qw

    if isinstance(q, np.ndarray):
        return np.array([t0, t1, t2, t3])
    else:
        return cs.vertcat(t0, t1, t2, t3)

def rotation_matrix_to_quat(rot):
    """
    Calculate a quaternion from a 3x3 rotation matrix.
    :param rot: 3x3 numpy array, representing a valid rotation matrix
    :return: a quaternion corresponding to the 3D rotation described by the input matrix. Quaternion format: wxyz
    """

    q = pyquaternion.Quaternion(matrix=rot)
    return np.array([q.w, q.x, q.y, q.z])

def undo_quaternion_flip(q_past, q_current):
    """
    Detects if q_current generated a quaternion jump and corrects it. Requires knowledge of the previous quaternion
    in the series, q_past
    :param q_past: 4-dimensional vector representing a quaternion in wxyz form.
    :param q_current: 4-dimensional vector representing a quaternion in wxyz form. Will be corrected if it generates
    a flip wrt q_past.
    :return: q_current with the flip removed if necessary
    """
    if np.sqrt(np.sum((q_past - q_current) ** 2)) > np.sqrt(np.sum((q_past + q_current) ** 2)):
        return -q_current
    return q_current


def discretize_dynamics_and_cost(t_horizon, n_points, m_steps_per_point, X, u, dynamics_f, cost_f):
    """
    Integrates the symbolic dynamics and cost equations until the time horizon using a RK4 method.
    :param t_horizon: time horizon in seconds
    :param n_points: number of control input points until time horizon
    :param m_steps_per_point: number of integrations steps per control input
    :param x: 4-element list with symbolic vectors for position (3D), angle (4D), velocity (3D) and rate (3D)
    :param u: 4-element symbolic vector for control input
    :param dynamics_f: symbolic dynamics function written in CasADi symbolic syntax.
    :param cost_f: symbolic cost function written in CasADi symbolic syntax. If None, then cost 0 is returned.
    :return: a symbolic function that computes the dynamics integration and the cost function at n_control_inputs
    points until the time horizon given an initial state and control
    """
    # Fixed step Runge-Kutta 4 integrator
    dt = t_horizon / n_points / m_steps_per_point
    X0 = X
    q = 0

    for j in range(m_steps_per_point):
        k1 = dynamics_f(X=X, u=u)['X_dot']
        k2 = dynamics_f(X=X + dt / 2 * k1, u=u)['X_dot']
        k3 = dynamics_f(X=X + dt / 2 * k2, u=u)['X_dot']
        k4 = dynamics_f(X=X + dt * k3, u=u)['X_dot']
        X_out = X + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        X = X_out
        q = q + cost_f(X=X, u=u)['q']

    return cs.Function('F', [X0, u], [X, q], ['X0', 'p'], ['Xf', 'qf'])

def activation(x, act):
    if act == 'relu':
        if isinstance(x, np.ndarray):
            return np.maximum(0,x)
        return cs.if_else(x < 0, 0.* x, x)
    if act == 'linear':
        return x
    if act =='elu':
        if isinstance(x, np.ndarray):
            return np.maximum(0,x)
        return cs.if_else(x < 0, -0.2 * x, x)