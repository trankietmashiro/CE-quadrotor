import numpy as np
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

from cem import run_cem
from mppi import run_mppi

# ── State layout ──
# [x, y, z, phi, theta, psi, xdot, ydot, zdot, phidot, thetadot, psidot]
nX = 12

# ── Initial / desired states ──
x0 = np.zeros(nX)
xd = np.zeros(nX)
xd[0], xd[1], xd[2] = 10.0, 10.0, 10.0


# ══════════════════════════════════════════════════════════════
#  Rotation helpers
# ══════════════════════════════════════════════════════════════

def _rot_z(psi):
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[ c, -s, 0],
                     [ s,  c, 0],
                     [ 0,  0, 1]])

def _rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0, s],
                     [ 0, 1, 0],
                     [-s, 0, c]])

def _rot_x(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0,  0],
                     [0, c, -s],
                     [0, s,  c]])

def rotation_matrix(phi, theta, psi):
    """ZYX Euler rotation matrix (body → world)."""
    return _rot_z(psi) @ _rot_y(theta) @ _rot_x(phi)


# ══════════════════════════════════════════════════════════════
#  Position plot
# ══════════════════════════════════════════════════════════════

def plot_results(xf, uOpt, dt, method_name=""):
    """Plot x, y, z positions and control inputs vs time."""
    num_steps = xf.shape[1]
    t_state = np.arange(num_steps) * dt
    t_ctrl  = np.arange(uOpt.shape[1]) * dt

    # ── Position plot ──
    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    labels = ["x", "y", "z"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for i, (ax, lbl, col) in enumerate(zip(axes, labels, colors)):
        ax.plot(t_state, xf[i], color=col, linewidth=1.4, label=lbl)
        ax.axhline(xd[i], color=col, linestyle="--", alpha=0.5,
                   label=f"{lbl}_d = {xd[i]}")
        ax.set_ylabel(f"{lbl} (m)")
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"{method_name} — Position vs Time", fontsize=13)
    fig.tight_layout()
    plt.savefig("position_plot.png", dpi=150)
    print("Saved position_plot.png")
    plt.show()

    # ── Control input plot ──
    fig2, axes2 = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    u_labels = ["Thrust", "τ_x", "τ_y", "τ_z"]
    u_colors = ["#d62728", "#9467bd", "#8c564b", "#e377c2"]
    u_units  = ["N", "N·m", "N·m", "N·m"]

    for i, (ax, lbl, col, unit) in enumerate(zip(axes2, u_labels, u_colors, u_units)):
        ax.plot(t_ctrl, uOpt[i], color=col, linewidth=1.2, label=lbl)
        ax.set_ylabel(f"{lbl} ({unit})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle(f"{method_name} — Control Inputs vs Time", fontsize=13)
    fig2.tight_layout()
    plt.savefig("control_plot.png", dpi=150)
    print("Saved control_plot.png")
    plt.show()


# ══════════════════════════════════════════════════════════════
#  3‑D quadrotor animation
# ══════════════════════════════════════════════════════════════

def animate_quadrotor(xf, dt, method_name="", L=0.17, speedup=4):
    """
    Animate the quadrotor as two crossed bars in 3‑D.

    The two arms are drawn in different colours so orientation is visible.
    A red star marks the goal and a translucent trail follows the quad.

    Parameters
    ----------
    xf       : state trajectory (nX, T)
    dt       : timestep (s)
    L        : arm half‑length (m)  — scaled up for visibility
    speedup  : playback speed multiplier
    """
    # Scale arm length for visibility (L=0.17 m is tiny on a 10 m plot)
    arm_scale = max(1.0, np.linalg.norm(xd[:3]) * 0.08)

    pos    = xf[:3]        # (3, T)
    angles = xf[3:6]       # phi, theta, psi  (3, T)

    # ── Arm endpoints in body frame ──
    arm1_body = np.array([[ arm_scale, 0, 0],
                          [-arm_scale, 0, 0]]).T       # (3, 2)
    arm2_body = np.array([[ 0,  arm_scale, 0],
                          [ 0, -arm_scale, 0]]).T       # (3, 2)

    # ── Figure ──
    fig = plt.figure(figsize=(9, 8))
    ax  = fig.add_subplot(111, projection="3d")

    all_pts = np.hstack([pos, xd[:3, None]])
    margin  = arm_scale + 1.5
    lo = all_pts.min(axis=1) - margin
    hi = all_pts.max(axis=1) + margin

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"{method_name} Quadrotor Animation")

    # ── Goal ──
    ax.scatter(*xd[:3], s=200, c="red", marker="*", zorder=5, label="Goal")
    ax.legend(loc="upper left", fontsize=9)

    # ── Artists ──
    trail_line, = ax.plot([], [], [], "-", color="steelblue",
                          linewidth=0.8, alpha=0.5)
    arm1_line,  = ax.plot([], [], [], "o-", color="royalblue",
                          linewidth=3.5, markersize=5, markerfacecolor="navy")
    arm2_line,  = ax.plot([], [], [], "o-", color="darkorange",
                          linewidth=3.5, markersize=5, markerfacecolor="saddlebrown")
    time_text   = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)

    # ── Frame sampling ──
    step = max(1, int(speedup))
    frame_indices = list(range(0, pos.shape[1], step))
    interval_ms = max(10, dt * step * 1000 / speedup)

    def _update(i):
        R = rotation_matrix(angles[0, i], angles[1, i], angles[2, i])
        p = pos[:, i]

        a1 = p[:, None] + R @ arm1_body
        a2 = p[:, None] + R @ arm2_body

        arm1_line.set_data(a1[0], a1[1]);  arm1_line.set_3d_properties(a1[2])
        arm2_line.set_data(a2[0], a2[1]);  arm2_line.set_3d_properties(a2[2])

        trail_line.set_data(pos[0, :i+1], pos[1, :i+1])
        trail_line.set_3d_properties(pos[2, :i+1])

        time_text.set_text(f"t = {i * dt:.2f} s")
        return arm1_line, arm2_line, trail_line, time_text

    anim = FuncAnimation(fig, _update, frames=frame_indices,
                         interval=interval_ms, blit=False, repeat=False)
    anim.save("quadrotor_animation.gif", writer="pillow", fps=10)
    print("Saved quadrotor_animation.gif")
    plt.show()
    return anim


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
    parser.add_argument("--plot",    action="store_true",       help="Show position plots")
    parser.add_argument("--animate", action="store_true",       help="Show 3-D animation")
    parser.add_argument("--speedup", type=float, default=4,     help="Animation speedup (default: 4)")
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
        animate_quadrotor(xf, dt, method_name=args.method.upper(),
                          speedup=args.speedup)
