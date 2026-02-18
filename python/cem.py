import numpy as np
from quadrotor import quadrotor

# ── Dynamics ──
DYNAMICS = quadrotor
nX = 12
nU = 4

# ── Default CEM parameters ──
DEFAULT_CEM_PARAMS = dict(
    num_samples = 1000,
    elite_frac  = 0.1,
    N           = 50,
    dt          = 0.1,
    alpha       = 0.2,
    eps_reg     = 1e-6,
    tau_limit   = 5e-3,
    min_var     = 1e-4,
    max_var     = 2.0,
    num_iters   = 100,
    base_cov    = np.diag([2.5, 5e-3, 5e-3, 5e-3]),
    Q           = np.diag([2.5, 2.5, 20.0, 1.0, 1.0, 15.0, 1, 1, 1, 1, 1, 1]),
    R           = np.diag([0.1, 0.1, 0.1, 0.1]),
    Qf          = 20.0 * np.diag([2.5, 2.5, 2.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0]),
)


def run_cem(x0, xd, params=None):
    """
    Run CEM trajectory optimisation.

    Parameters
    ----------
    x0     : initial state (nX,)
    xd     : desired state (nX,)
    params : dict of parameters (any missing key uses DEFAULT_CEM_PARAMS)

    Returns
    -------
    xf   : state trajectory  (nX, num_iters)
    uOpt : applied controls   (nU, num_iters)
    """
    p = {**DEFAULT_CEM_PARAMS, **(params or {})}

    num_samples = p["num_samples"]
    elite_frac  = p["elite_frac"]
    num_elite   = int(num_samples * elite_frac)
    N           = p["N"]
    dt          = p["dt"]
    alpha       = p["alpha"]
    eps_reg     = p["eps_reg"]
    tau_limit   = p["tau_limit"]
    min_var     = p["min_var"]
    max_var     = p["max_var"]
    num_iters   = p["num_iters"]
    base_cov    = p["base_cov"]
    Q           = p["Q"]
    R           = p["R"]
    Qf          = p["Qf"]

    du_max = np.array([10, tau_limit, tau_limit, tau_limit])[:, None]

    # Initial control distribution
    utraj = np.zeros((nU, N - 1))
    covu  = np.repeat(base_cov[:, :, None], N - 1, axis=2)

    x = x0.copy()
    xf_list   = []
    uOpt_list = []

    for iteration in range(num_iters):
        xf_list.append(x.copy())

        Straj = np.zeros(num_samples)
        Uall  = np.zeros((num_samples, nU, N - 1))

        # ── Rollouts ──
        for k in range(num_samples):
            eps = np.clip(np.random.randn(nU, N - 1), -3.0, 3.0)
            du  = np.einsum('ijt,jt->it', covu, eps)
            du  = np.clip(du, -du_max, du_max)

            u = utraj + du
            u[1:4, :] = np.clip(u[1:4, :], -tau_limit, tau_limit)
            Uall[k] = u

            xtraj = np.zeros((nX, N))
            xtraj[:, 0] = x
            valid = True

            for t in range(N - 1):
                u_t    = u[:, t]
                x_next = xtraj[:, t] + DYNAMICS(xtraj[:, t], u_t) * dt

                if not np.isfinite(x_next).all():
                    Straj[k] = 1e12
                    valid = False
                    break

                xtraj[:, t + 1] = x_next
                Straj[k] += (xtraj[:, t] - xd) @ Q @ (xtraj[:, t] - xd) + u_t @ R @ u_t

            if valid:
                Straj[k] += (xtraj[:, -1] - xd) @ Qf @ (xtraj[:, -1] - xd)

        # ── Elite selection ──
        elite_idx = np.argsort(Straj)[:num_elite]
        elite_U   = Uall[elite_idx]

        print(f"[CEM] Iter {iteration:3d} | minS = {np.min(Straj):.4f}", end="")

        # ── Update mean ──
        utraj_new = np.mean(elite_U, axis=0)

        # ── Update covariance (per timestep, diagonal) ──
        covu_new = np.zeros_like(covu)
        for t in range(N - 1):
            Ut  = elite_U[:, :, t]
            cov = np.cov(Ut, rowvar=False) + eps_reg * np.eye(nU)
            diag = np.clip(np.diag(cov), min_var, max_var)
            covu_new[:, :, t] = np.diag(diag)

        # ── Smooth update ──
        utraj = (1 - alpha) * utraj + alpha * utraj_new
        covu  = (1 - alpha) * covu  + alpha * covu_new

        # ── Apply first control & advance ──
        uOpt_list.append(utraj[:, 0].copy())
        x = x + DYNAMICS(x, utraj[:, 0]) * dt

        # ── Shift horizon ──
        utraj[:, :-1]   = utraj[:, 1:]
        utraj[:, -1]    = 0.0
        covu[:, :, :-1] = covu[:, :, 1:]
        covu[:, :, -1]  = base_cov

        distance = np.linalg.norm(x[:3] - xd[:3])
        print(f" | distance = {distance:.4f}")

    xf   = np.array(xf_list).T
    uOpt = np.array(uOpt_list).T
    return xf, uOpt
