# Quadrotor Trajectory Optimisation

Sampling-based trajectory optimisation for a 6-DOF quadrotor using **Cross-Entropy Method (CEM)** and **Model Predictive Path Integral (MPPI)** control in a receding-horizon framework.

## How It Works

The quadrotor is modelled with 12 states (position, Euler angles, and their rates) and 4 control inputs (collective thrust and body-axis torques). At each timestep, both planners:

1. **Sample** 1000 candidate control sequences from the current distribution.
2. **Evaluate** each by rolling it out through the dynamics and computing a quadratic cost on tracking error and control effort.
3. **Update** the nominal trajectory — CEM refits a Gaussian to elite samples; MPPI uses importance-weighted averaging across all rollouts.
4. **Apply** the first control, advance the state, and shift the horizon forward.

## Project Structure

```
main.py          Entry point, CLI, position plots, and 3-D animation
cem.py           CEM controller and default parameters
mppi.py          MPPI controller and default parameters
quadrotor.py     Quadrotor dynamics (external dependency)
```

## Usage

```bash
# CEM with default settings
python main.py cem

# MPPI with plots and animation
python main.py mppi --plot --animate

# Override parameters
python main.py cem --iters 300 --samples 2000 --horizon 250

# Faster animation playback
python main.py cem --animate --speedup 8
```

### CLI Options

| Flag | Description |
|------|-------------|
| `method` | `cem` or `mppi` (default: `cem`) |
| `--iters` | Number of MPC iterations |
| `--samples` | Rollout samples per iteration |
| `--horizon` | Planning horizon length N |
| `--dt` | Integration timestep (s) |
| `--lam` | MPPI temperature λ |
| `--alpha` | CEM smoothing factor α |
| `--plot` | Show position-vs-time plots |
| `--animate` | Show 3-D quadrotor animation |
| `--speedup` | Animation playback multiplier (default: 4) |

## Visualisation

- **Position plot** (`--plot`) — x, y, z vs time with dashed goal lines. Saved as `position_plot.png`.
- **3-D animation** (`--animate`) — Two coloured crossed arms rotate with body orientation; a red star marks the goal and a trail traces the flight path. Saved as `quadrotor_animation.gif`.

## Dependencies

- Python 3.8+
- NumPy
- Matplotlib
