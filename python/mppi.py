import numpy as np
from quadrotor import quadrotor

# ── Dynamics ──
DYNAMICS = quadrotor
nX = 12
nU = 4

# ── Default MPPI parameters ──
DEFAULT_MPPI_PARAMS = dict(
    num_samples = 1000,
    N           = 30,
    dt          = 0.1,
    lam         = 10.0,
    nu          = 1000.0,
    num_iters   = 150,
    base_cov    = np.diag([2.5, 5e-3, 5e-3, 5e-3]),
    Q           = np.diag([2.5, 2.5, 20.0, 1.0, 1.0, 15.0, 0, 0, 0, 0, 0, 0]),
    Qf          = 20.0 * np.diag([2.5, 2.5, 20.0, 1.0, 1.0, 15.0, 0, 0, 0, 0, 0, 0]),
)


def run_mppi(x0, xd, params=None):
    """
    Run MPPI trajectory optimisation.

    Parameters
    ----------
    x0     : initial state (nX,)
    xd     : desired state (nX,)
    params : dict of parameters (any missing key uses DEFAULT_MPPI_PARAMS)

    Returns
    -------
    xf   : state trajectory  (nX, num_iters)
    uOpt : applied controls   (nU, num_iters)
    """
    p = {**DEFAULT_MPPI_PARAMS, **(params or {})}

    num_samples = p["num_samples"]
    N           = p["N"]
    dt          = p["dt"]
    lam         = p["lam"]
    nu          = p["nu"]
    num_iters   = p["num_iters"]
    base_cov    = p["base_cov"]
    Q           = p["Q"]
    Qf          = p["Qf"]

    R_cost = lam * np.linalg.inv(base_cov)

    utraj = np.zeros((nU, N - 1))
    x = x0.copy()
    xf_list   = []
    uOpt_list = []

    for iteration in range(num_iters):
        xf_list.append(x.copy())

        Straj = np.zeros(num_samples)
        dU = []

        # ── Rollouts ──
        for k in range(num_samples):
            du = base_cov @ np.random.randn(nU, N - 1)
            dU.append(du)
            xtraj = np.zeros((nX, N))
            xtraj[:, 0] = x

            for t in range(N - 1):
                u_t = utraj[:, t]
                xtraj[:, t + 1] = xtraj[:, t] + DYNAMICS(xtraj[:, t], u_t + du[:, t]) * dt

                qx = (xtraj[:, t] - xd) @ Q @ (xtraj[:, t] - xd)
                Straj[k] += (qx
                             + 0.5 * u_t @ R_cost @ u_t
                             + (1 - 1 / nu) / 2 * du[:, t] @ R_cost @ du[:, t]
                             + u_t @ R_cost @ du[:, t])

            Straj[k] += (xtraj[:, N - 1] - xd) @ Qf @ (xtraj[:, N - 1] - xd)

        minS = np.min(Straj)
        print(f"[MPPI] Iter {iteration:3d} | minS = {minS:.4f}", end="  ")

        # ── Update nominal inputs ──
        weights = np.exp(-1.0 / lam * (Straj - minS))
        for t in range(N - 1):
            du_all = np.array([dU[k][:, t] for k in range(num_samples)])
            utraj[:, t] += (weights @ du_all) / np.sum(weights)

        # ── Execute first control ──
        uOpt_list.append(utraj[:, 0].copy())
        x = x + DYNAMICS(x, utraj[:, 0]) * dt

        # ── Shift nominal inputs ──
        utraj[:, :-1] = utraj[:, 1:]
        utraj[:, -1]  = 0.0

        distance = np.linalg.norm(x[:3] - xd[:3])
        print(f"| distance = {distance:.4f}")

    xf   = np.array(xf_list).T
    uOpt = np.array(uOpt_list).T
    return xf, uOpt
