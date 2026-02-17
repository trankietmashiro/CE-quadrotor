import numpy as np


def quadrotor(s, u):
    """
    Quadrotor dynamics.

    Parameters
    ----------
    s : array-like, shape (12,)
        State vector: [x, y, z, roll, pitch, yaw, vx, vy, vz, p, q, r]
    u : array-like, shape (4,)
        Input vector: [thrust, tx, ty, tz]

    Returns
    -------
    f : ndarray, shape (12,)
        State derivative.
    """
    m = 0.2
    g = 9.81
    J = np.diag([7e-3, 7e-3, 12e-3])
    e3 = np.array([0.0, 0.0, 1.0])

    # Unpack state
    x, y, z = s[0], s[1], s[2]
    roll, pitch, yaw = s[3], s[4], s[5]
    vx, vy, vz = s[6], s[7], s[8]
    p, q, r = s[9], s[10], s[11]

    thrust = u[0]
    tx, ty, tz = u[1], u[2], u[3]

    # Position derivatives
    xdot = vx
    ydot = vy
    zdot = vz

    # Rotation matrix
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R = np.array([
        [cy * cp - sr * sy * sp, -cr * sy, cy * sp + cp * sr * sy],
        [cp * sy + cy * sr * sp,  cr * cy, sy * sp - cy * cp * sr],
        [-cr * sp,                sr,      cr * cp]
    ])

    # Velocity derivatives
    vdot = R @ (thrust / m * e3) - g * e3
    vxdot, vydot, vzdot = vdot[0], vdot[1], vdot[2]

    # Euler rate transformation
    T = np.array([
        [cp,  0.0, -cr * sp],
        [0.0, 1.0,  sr],
        [sp,  0.0,  cr * cp]
    ])

    omega_body = np.array([p, q, r])
    w = np.linalg.solve(T, omega_body)
    rolldot, pitchdot, yawdot = w[0], w[1], w[2]

    # Angular acceleration
    wdot = np.linalg.solve(J, np.array([tx, ty, tz]) - np.cross(omega_body, J @ omega_body))
    pdot, qdot, rdot = wdot[0], wdot[1], wdot[2]

    f = np.array([xdot, ydot, zdot, rolldot, pitchdot, yawdot,
                  vxdot, vydot, vzdot, pdot, qdot, rdot])
    return f