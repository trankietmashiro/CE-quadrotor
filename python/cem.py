import numpy as np
from quadrotor import quadrotor

# =====================================================
# Dynamics
# =====================================================
DYNAMICS = quadrotor
nX = 12
nU = 4

# =====================================================
# Default CEM parameters
# =====================================================
DEFAULT_CEM_PARAMS = dict(
    num_samples = 1000,
    elite_frac  = 0.01,
    N           = 50,
    dt          = 0.1,
    alpha       = 0.4,
    eps_reg     = 1e-6,
    tau_limit   = 5e-3,

    # ---- variance limits (PER CONTROL) ----
    min_var     = np.array([0, 0, 0, 0]),
    max_var     = 5.0,

    num_iters   = 50,       # CEM refinement iterations

    # base exploration scale
    base_cov    = np.diag([2.5, 5e-3, 5e-3, 5e-3]),

    # cost weights
    Q  = np.diag([0, 0, 0, 0.0, 0.0, 0.0,
                  0, 0, 0, 0, 0, 0]),
    R  = np.diag([0., 0., 0., 0.]),
    Qf = 1000.0 * np.diag([2.5, 2.5, 2.5, 0.0, 0.0, 0.0,
                         0, 0, 0, 0, 0, 0]),
)


# =====================================================
# CEM Optimizer  (open-loop, no horizon shift)
# =====================================================
def run_cem(x0, xd, params=None):
    """
    Run CEM trajectory optimisation over a fixed horizon.

    The optimizer refines a single N-step control sequence
    over `num_iters` CEM iterations, then simulates the
    full trajectory with the best controls found.

    Parameters
    ----------
    x0 : (nX,)   initial state
    xd : (nX,)   desired state

    Returns
    -------
    xtraj : state trajectory   (nX, N)
    utraj : control trajectory (nU, N-1)
    """

    # merge parameters
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

    # control increment limits
    du_max = np.array([10, tau_limit, tau_limit, tau_limit])[:, None]

    # =====================================================
    # Initial control distribution
    # =====================================================
    utraj_mean = np.zeros((nU, N - 1))
    covu = np.repeat(base_cov[:, :, None], N - 1, axis=2)   # (nU, nU, N-1)

    # =====================================================
    # CEM refinement loop
    # =====================================================
    for iteration in range(num_iters):

        Straj = np.zeros(num_samples)
        Uall  = np.zeros((num_samples, nU, N - 1))

        # -------------------------------------------------
        # Sample & rollout
        # -------------------------------------------------
        for k in range(num_samples):

            eps = np.clip(
                np.random.randn(nU, N - 1),
                -3.0, 3.0
            )

            du = np.einsum('ijt,jt->it', covu, eps)
            du = np.clip(du, -du_max, du_max)

            u = utraj_mean + du
            u[1:4, :] = np.clip(u[1:4, :], -tau_limit, tau_limit)

            Uall[k] = u

            xtraj = np.zeros((nX, N))
            xtraj[:, 0] = x0

            valid = True

            for t in range(N - 1):
                u_t = u[:, t]
                x_next = xtraj[:, t] + DYNAMICS(xtraj[:, t], u_t) * dt

                if not np.isfinite(x_next).all():
                    Straj[k] = 1e12
                    valid = False
                    break

                xtraj[:, t + 1] = x_next

                dx = xtraj[:, t] - xd
                Straj[k] += dx @ Q @ dx + u_t @ R @ u_t

            if valid:
                dx = xtraj[:, -1] - xd
                Straj[k] += dx @ Qf @ dx

        # -------------------------------------------------
        # Elite selection
        # -------------------------------------------------
        elite_idx = np.argsort(Straj)[:num_elite]
        elite_U   = Uall[elite_idx]

        best_cost = Straj[elite_idx[0]]
        print(
            f"[CEM] Iter {iteration:3d} | "
            f"minS = {np.min(Straj):.4f} | "
            f"elite mean = {np.mean(Straj[elite_idx]):.4f}"
        )

        # -------------------------------------------------
        # Mean update
        # -------------------------------------------------
        utraj_new = np.mean(elite_U, axis=0)

        # -------------------------------------------------
        # Covariance update (DIAGONAL)
        # -------------------------------------------------
        covu_new = np.zeros_like(covu)

        for t in range(N - 1):
            Ut  = elite_U[:, :, t]
            cov = np.cov(Ut, rowvar=False) + eps_reg * np.eye(nU)

            diag = np.diag(cov)
            diag = np.maximum(diag, min_var)
            diag = np.minimum(diag, max_var)

            covu_new[:, :, t] = np.diag(diag)

        # -------------------------------------------------
        # Smooth update
        # -------------------------------------------------
        utraj_mean = (1 - alpha) * utraj_mean + alpha * utraj_new
        covu       = (1 - alpha) * covu       + alpha * covu_new

    # =====================================================
    # Simulate final trajectory with optimised controls
    # =====================================================
    xtraj = np.zeros((nX, N))
    xtraj[:, 0] = x0

    for t in range(N - 1):
        u_t = utraj_mean[:, t]
        xtraj[:, t + 1] = xtraj[:, t] + DYNAMICS(xtraj[:, t], u_t) * dt

    distance = np.linalg.norm(xtraj[:3, -1] - xd[:3])
    print(f"\n[CEM] Done — final distance = {distance:.4f}")

    return xtraj, utraj_mean
