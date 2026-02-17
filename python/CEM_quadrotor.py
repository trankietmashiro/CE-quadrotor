import numpy as np
from quadrotor import quadrotor

# ── Cost functions ──

def running_cost(x, xd, u):
    Q = np.diag([2.5, 2.5, 20.0, 1.0, 1.0, 15.0, 1, 1, 1, 1, 1, 1])
    R = np.diag([0.1, 0.1, 0.1, 0.1])
    return (x - xd) @ Q @ (x - xd) + u @ R @ u


def final_cost(xT, xd):
    Qf = 20.0 * np.diag([2.5, 2.5, 2.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0])
    return (xT - xd) @ Qf @ (xT - xd)


# ── Dynamics ──

DYNAMICS = quadrotor

nX = 12
nU = 4

# Quadrotor w² to force/torque matrix
kf = 8.55e-6 * 91.61
L  = 0.17
b  = 1.6e-2 * 91.61
m  = 0.716
g  = 9.81

A = np.array([
    [ kf,    kf,   kf,    kf   ],
    [ 0,   L*kf,   0,  -L*kf  ],
    [-L*kf,  0,   L*kf,   0   ],
    [ b,    -b,    b,    -b   ],
])

# ── Initial / desired states ──

x0 = np.zeros(nX)
xd = np.zeros(nX)
xd[0], xd[1], xd[2] = 10.0, 10.0, 10.0

# ── CEM parameters ──

num_samples = 1000
elite_frac  = 0.1
num_elite   = int(num_samples * elite_frac)
N           = 200          # planning horizon
dt          = 0.02
alpha       = 0.2          # smoothing factor
eps_reg     = 1e-6         # covariance regularisation
tau_limit   = 5e-3         # torque bound
min_var     = 1e-4
max_var     = 2.0

du_max = np.array([10, tau_limit, tau_limit, tau_limit])[:, None]

# ── Initial control distribution  ──
#    utraj : (nU, N-1)
#    covu  : (nU, nU, N-1)

utraj    = np.zeros((nU, N - 1))
base_cov = np.diag([2.5, 5e-3, 5e-3, 5e-3])
covu     = np.repeat(base_cov[:, :, None], N - 1, axis=2)   # (nU, nU, N-1)

# ── Main loop ──

x = x0.copy()
xf_list   = []
uOpt_list = []

for iteration in range(200):

    xf_list.append(x.copy())

    Straj = np.zeros(num_samples)
    Uall  = np.zeros((num_samples, nU, N - 1))

    # ── Rollouts ──
    for k in range(num_samples):

        # sample perturbation
        eps = np.clip(np.random.randn(nU, N - 1), -3.0, 3.0)

        # du[:, t] = covu[:, :, t] @ eps[:, t]
        du = np.einsum('ijt,jt->it', covu, eps)
        du = np.clip(du, -du_max, du_max)

        u = utraj + du
        u[1:4, :] = np.clip(u[1:4, :], -tau_limit, tau_limit)
        Uall[k] = u

        # simulate
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
            Straj[k] += running_cost(xtraj[:, t], xd, u_t)

        if valid:
            Straj[k] += final_cost(xtraj[:, -1], xd)

    # ── Elite selection ──
    elite_idx = np.argsort(Straj)[:num_elite]
    elite_U   = Uall[elite_idx]                    # (num_elite, nU, N-1)

    print(f"Iter {iteration:3d} | minS = {np.min(Straj):.4f}", end="")

    # ── Update mean ──
    utraj_new = np.mean(elite_U, axis=0)           # (nU, N-1)

    # ── Update covariance (per timestep) ──
    covu_new = np.zeros_like(covu)                  # (nU, nU, N-1)

    for t in range(N - 1):
        Ut  = elite_U[:, :, t]                      # (num_elite, nU)
        cov = np.cov(Ut, rowvar=False) + eps_reg * np.eye(nU)

        # clip variances, keep diagonal
        diag = np.clip(np.diag(cov), min_var, max_var)
        covu_new[:, :, t] = np.diag(diag)

    # ── Smooth update ──
    utraj = (1 - alpha) * utraj + alpha * utraj_new
    covu  = (1 - alpha) * covu  + alpha * covu_new

    # ── Apply first control & advance ──
    x = x + DYNAMICS(x, utraj[:, 0]) * dt

    # ── Shift horizon ──
    utraj[:, :-1]    = utraj[:, 1:]
    utraj[:, -1]     = 0.0

    covu[:, :, :-1]  = covu[:, :, 1:]
    covu[:, :, -1]   = base_cov

    distance = np.linalg.norm(x[:3] - xd[:3])
    print(f" | distance = {distance:.4f}")