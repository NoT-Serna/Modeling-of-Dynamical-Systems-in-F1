import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------
# PHYSICAL CONSTANTS
# --------------------------------------------------------
m = 605                  # mass (kg)
mu = 1.7                 # friction coefficient (soft tires)
g = 9.81                 # gravity
Cd = 0.8                 # lumped drag coefficient
P_max = 900 * 745.7      # engine power (900hp → watts)


# --------------------------------------------------------
# TRACK MODEL (Interlagos – simplified)
# Defined as 15 segments with length and curvature
# Curvature kappa = 1/R (positive left, negative right)
# --------------------------------------------------------
segments = [
    (300, 0.0),    # straight
    (120, 1/70),   # left turn
    (200, 0.0),
    (150, -1/60),  # right turn
    (250, 0.0),
    (180, 1/80),   # left
    (300, 0.0),
    (200, -1/90),  # right
    (100, 0.0),
    (160, 1/60),
    (350, 0.0),
    (140, -1/50),
    (260, 0.0),
    (110, 1/80),
    (289, 0.0)
]

# Expand into discrete curvature array along the lap
ds = 1.0  # spatial step (meters)
kappa = []
for length, curv in segments:
    steps = int(length / ds)
    kappa += [curv] * steps

kappa = np.array(kappa)
L = len(kappa)


# --------------------------------------------------------
# SIMULATION PARAMETERS
# --------------------------------------------------------
v = 5.0                  # start at 5 m/s
velocity = []

# Full throttle everywhere
u = 1.0

for i in range(L):

    F_engine = (u * P_max) / max(v, 1e-3)
    F_drag   = Cd * v**2
    F_lat    = m * v**2 * kappa[i]

    F_max = mu * m * g
    F_long_max = np.sqrt(max(F_max**2 - F_lat**2, 0))

    F_long_desired = F_engine - F_drag
    F_long = np.clip(F_long_desired, -F_long_max, F_long_max)

    dv = (F_long / m) * ds
    v = max(v + dv, 0)

    velocity.append(v)


velocity = np.array(velocity)


# --------------------------------------------------------
# GRIP USAGE CALCULATION
# --------------------------------------------------------
F_lat_array = m * velocity**2 * kappa
F_max = mu * m * g
grip_usage = np.clip(np.abs(F_lat_array) / F_max, 0, 1)


# --------------------------------------------------------
# PLOT VELOCITY
# --------------------------------------------------------
s_axis = np.arange(L) * ds

plt.figure(figsize=(12,5))
plt.plot(s_axis, velocity, linewidth=2)
plt.title("Velocity Profile Along Interlagos")
plt.xlabel("Track Position s (m)")
plt.ylabel("Velocity (m/s)")
plt.grid()
plt.show()


# --------------------------------------------------------
# PLOT CURVATURE
# --------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(s_axis, kappa, linewidth=2)
plt.title("Track Curvature (κ)")
plt.xlabel("Track Position s (m)")
plt.ylabel("Curvature (1/m)")
plt.grid()
plt.show()


# --------------------------------------------------------
# PLOT GRIP USAGE WITH TURN MARKERS
# --------------------------------------------------------
plt.figure(figsize=(12,5))
plt.plot(s_axis, grip_usage*100, linewidth=2, label="Grip Usage (%)")

for i in range(1, L):
    if kappa[i] != 0 and kappa[i-1] == 0:
        plt.axvline(s_axis[i], color='r', linestyle='--', alpha=0.4)

plt.title("Grip Usage Along the Lap")
plt.xlabel("Track Position s (m)")
plt.ylabel("Grip Usage (%)")
plt.ylim(0, 110)
plt.grid()
plt.show()


# --------------------------------------------------------
# SEGMENT STABILITY REPORT
# --------------------------------------------------------
print("\nSEGMENT STABILITY REPORT")
print("----------------------------")

index = 0
for seg_id, (length, curv) in enumerate(segments, start=1):
    steps = int(length / ds)
    seg_grip = grip_usage[index:index+steps]
    avg_grip = np.mean(seg_grip)

    if curv == 0:
        type_str = "Straight"
    else:
        type_str = f"Curve (κ={curv:.4f})"

    stability_score = (1 - avg_grip) * 100

    print(f"Segment {seg_id:2d}: {type_str:18s}  |  Avg Grip Use = {avg_grip*100:5.1f}%  |  Stability Score = {stability_score:5.1f}%")

    index += steps

