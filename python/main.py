import numpy as np
import argparse

from cem import run_cem
from mppi import run_mppi

# ── Initial / desired states ──
nX = 12

x0 = np.zeros(nX)
xd = np.zeros(nX)
xd[0], xd[1], xd[2] = 10.0, 10.0, 10.0


# ══════════════════════════════════════════════════════════════
#  Plotting / Animation  (TODO)
# ══════════════════════════════════════════════════════════════

def plot_results(xf, uOpt, dt, method_name=""):
    """Plot state trajectories and control inputs."""
    # TODO: add figures here
    pass


def animate_quadrotor(xf, dt, method_name=""):
    """3‑D animation of the quadrotor trajectory."""
    # TODO: add animation here
    pass


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quadrotor trajectory optimisation")
    parser.add_argument(
        "method",
        nargs="?",
        default="cem",
        choices=["cem", "mppi"],
        help="Optimisation method: 'cem' or 'mppi' (default: cem)",
    )
    parser.add_argument("--iters",   type=int,   default=None, help="Number of iterations")
    parser.add_argument("--samples", type=int,   default=None, help="Number of samples")
    parser.add_argument("--horizon", type=int,   default=None, help="Planning horizon N")
    parser.add_argument("--dt",      type=float, default=None, help="Timestep")
    parser.add_argument("--lam",     type=float, default=None, help="MPPI temperature lambda")
    parser.add_argument("--alpha",   type=float, default=None, help="CEM smoothing factor")
    parser.add_argument("--plot",    action="store_true",       help="Show result plots")
    parser.add_argument("--animate", action="store_true",       help="Show 3‑D animation")
    args = parser.parse_args()

    # Build override dict from CLI args
    overrides = {}
    if args.iters   is not None: overrides["num_iters"]  = args.iters
    if args.samples is not None: overrides["num_samples"] = args.samples
    if args.horizon is not None: overrides["N"]           = args.horizon
    if args.dt      is not None: overrides["dt"]          = args.dt
    if args.lam     is not None: overrides["lam"]         = args.lam
    if args.alpha   is not None: overrides["alpha"]       = args.alpha

    dt = overrides.get("dt", 0.02)

    if args.method == "cem":
        xf, uOpt = run_cem(x0, xd, params=overrides)
    else:
        xf, uOpt = run_mppi(x0, xd, params=overrides)

    print(f"\nDone. xf shape: {xf.shape}, uOpt shape: {uOpt.shape}")

    if args.plot:
        plot_results(xf, uOpt, dt, method_name=args.method.upper())

    if args.animate:
        animate_quadrotor(xf, dt, method_name=args.method.upper())
