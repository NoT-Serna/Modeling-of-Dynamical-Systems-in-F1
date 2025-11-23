# Phase portrait and time-series for the FW26 longitudinal model
import math
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------
# PARAMETERS (FW26-like)
# -------------------------
m = 605.0                   # kg
P_max = 670000.0            # W (≈900 bhp)
rho = 1.225                 # kg/m^3
Cd = 0.8
A = 1.55                    # m^2
Cd_p = 0.5 * rho * Cd * A   # lumped drag coefficient
F_brake_max = m * 5 * 9.81  # brake estimate (not used for u=1)
eps = 1e-3                  # regularizer to avoid division by zero
u = 1.0                     # constant throttle (full)

# -------------------------
# FORCE MODELS
# -------------------------
def F_engine(u, v):
    # power-limited engine (regularized)
    return u * P_max / max(v, eps)

def F_drag(v):
    return Cd_p * v**2

def F_brake(u):
    # simple piecewise brake model (not used for positive u)
    return 0.0 if u >= 0 else -abs(u) * F_brake_max

# -------------------------
# DYNAMICS (ODE)
# x = [s, v]
# -------------------------
def dynamics(t, x):
    s, v = x
    v_eff = max(v, 0.0)            # keep physics sensible
    F_e = F_engine(u, v_eff)
    F_d = F_drag(v_eff)
    F_b = F_brake(u)
    dv = (F_e - F_d - F_b) / m
    ds = v
    return [ds, dv]

# -------------------------
# EQUILIBRIUM SPEED (numerical)
# solve u*P/(v+eps) - Cd_p*v^2 = 0
# -------------------------
def equilibrium_speed(u):
    def g(v):
        return (u * P_max) / max(v, eps) - Cd_p * v**2
    a, b = 0.1, 300.0
    fa, fb = g(a), g(b)
    if fa * fb > 0:
        # fallback: return midpoint if no sign change
        return 0.5*(a+b)
    for _ in range(100):
        c = 0.5*(a+b)
        fc = g(c)
        if fa * fc <= 0:
            b, fb = c, fc
        else:
            a, fa = c, fc
    return 0.5*(a+b)

v_star = equilibrium_speed(u)
print(f"Computed equilibrium speed v* = {v_star:.2f} m/s  ({v_star*3.6:.1f} km/h)")

# -------------------------
# SIMULATE several initial speeds
# -------------------------
t_span = (0, 60)                     # seconds
t_eval = np.linspace(t_span[0], t_span[1], 2000)
initial_vs = [0.1, 20.0, 50.0, 80.0, 110.0]

trajectories = []
for v0 in initial_vs:
    x0 = [0.0, v0]
    sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval, max_step=0.1, rtol=1e-6)
    trajectories.append(sol)

# -------------------------
# PLOT 1: phase portrait v(s)
# -------------------------
plt.figure(figsize=(9,5))
for sol, v0 in zip(trajectories, initial_vs):
    s_vals = sol.y[0]
    v_vals = sol.y[1]
    plt.plot(s_vals, v_vals, label=f"v0={v0:.1f} m/s")
plt.axhline(y=v_star, linestyle='--', linewidth=1)
plt.text(0.02*np.max(trajectories[-1].y[0]), v_star+1, f"v* = {v_star:.1f} m/s", va='bottom')
plt.xlabel("Position along track s [m]")
plt.ylabel("Velocity v [m/s]")
plt.title("Phase portrait: Velocity vs Position (v(s))")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------
# PLOT 2: time series v(t)
# -------------------------
plt.figure(figsize=(9,5))
for sol, v0 in zip(trajectories, initial_vs):
    t_vals = sol.t
    v_vals = sol.y[1]
    plt.plot(t_vals, v_vals, label=f"v0={v0:.1f} m/s")
plt.axhline(y=v_star, linestyle='--', linewidth=1)
plt.text(0.02*(t_span[1]-t_span[0]), v_star+1, f"v* = {v_star:.1f} m/s", va='bottom')
plt.xlabel("Time t [s]")
plt.ylabel("Velocity v [m/s]")
plt.title("Velocity vs Time (v(t)) for several initial speeds")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
