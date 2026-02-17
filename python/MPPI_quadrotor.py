import numpy as np
from quadrotor import quadrotor

# ── Dynamics handle ──
DYNAMICS = quadrotor

nX = 12  # number of states
nU = 4   # number of inputs

# Quadrotor w² to force/torque matrix
kf = 8.55e-6 * 91.61
L = 0.17
b = 1.6e-2 * 91.61
m = 0.716
g = 9.81

A = np.array([
    [kf,     kf,   kf,    kf],
    [0,    L*kf,    0, -L*kf],
    [-L*kf,  0,  L*kf,     0],
    [b,     -b,    b,     -b]
])

# ── Initial and desired states ──
x0 = np.zeros(nX)
xd = np.zeros(nX)
xd[0], xd[1], xd[2] = 10.0, 10.0, 10.0

# ── MPPI parameters ──
num_samples = 1000
N = 150

utraj = np.zeros((nU, N - 1))
uOpt_list = []
xf_list = []
dt = 0.02
lam = 10.0        # lambda
nu = 1000.0
covu = np.diag([2.5, 5e-3, 5e-3, 5e-3])

R = lam * np.linalg.inv(covu)

x = x0.copy()


# ── Helper functions ──
def running_cost(x, xd, R, u, du, nu):
    Q = np.diag([2.5, 2.5, 20.0, 1.0, 1.0, 15.0, 0, 0, 0, 0, 0, 0])
    qx = (x - xd) @ Q @ (x - xd)
    J = qx + 0.5 * u @ R @ u + (1 - 1 / nu) / 2 * du @ R @ du + u @ R @ du
    return J


def final_cost(xT, xd):
    Qf = 20.0 * np.diag([2.5, 2.5, 20.0, 1.0, 1.0, 15.0, 0, 0, 0, 0, 0, 0])
    return (xT - xd) @ Qf @ (xT - xd)


# ── Run MPPI Optimization ──
for iteration in range(500):
    xf_list.append(x.copy())
    Straj = np.zeros(num_samples)
    dU = []

    # Rollouts
    for k in range(num_samples):
        du = covu @ np.random.randn(nU, N - 1)
        dU.append(du)
        xtraj = np.zeros((nX, N))
        xtraj[:, 0] = x

        for t in range(N - 1):
            u_t = utraj[:, t]
            xtraj[:, t + 1] = xtraj[:, t] + DYNAMICS(xtraj[:, t], u_t + du[:, t]) * dt
            Straj[k] += running_cost(xtraj[:, t], xd, R, u_t, du[:, t], nu)

        Straj[k] += final_cost(xtraj[:, N - 1], xd)

    minS = np.min(Straj)
    print(f"Iter {iteration:3d} | minS = {minS:.4f}", end="  ")

    # Update nominal inputs
    weights = np.exp(-1.0 / lam * (Straj - minS))
    for t in range(N - 1):
        # Gather perturbations for this timestep across all samples
        du_all = np.array([dU[k][:, t] for k in range(num_samples)])  # (num_samples, nU)
        utraj[:, t] += (weights @ du_all) / np.sum(weights)

    # Execute first control
    x = x + DYNAMICS(x, utraj[:, 0]) * dt
    uOpt_list.append(utraj[:, 0].copy())

    # Shift nominal inputs
    utraj[:, :-1] = utraj[:, 1:]
    utraj[:, -1] = 0.0

    distance = np.linalg.norm(x[:3] - xd[:3])
    print(f"| distance = {distance:.4f}")

# Convert logged data to arrays
xf = np.array(xf_list).T   # shape (nX, num_iters)
uOpt = np.array(uOpt_list).T  # shape (nU, num_iters)