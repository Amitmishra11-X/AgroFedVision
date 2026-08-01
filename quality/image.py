import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------

T = 1000

beta_start = 0.0001
beta_end = 0.02

# ----------------------------
# Linear Schedule
# ----------------------------

beta_linear = np.linspace(beta_start, beta_end, T)

alpha_linear = 1 - beta_linear

alpha_bar_linear = np.cumprod(alpha_linear)

# ----------------------------
# Cosine Schedule
# ----------------------------

s = 0.008

steps = np.arange(T + 1)

f = np.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2

alpha_bar_cosine = f / f[0]

# Remove the last point for equal length
alpha_bar_cosine = alpha_bar_cosine[:-1]

# ----------------------------
# Plot
# ----------------------------

plt.figure(figsize=(8,5))

plt.plot(
    alpha_bar_linear,
    linewidth=3,
    label="Linear Schedule"
)

plt.plot(
    alpha_bar_cosine,
    linewidth=3,
    label="Cosine Schedule"
)

plt.xlabel("Diffusion Timestep", fontsize=12)

plt.ylabel("Remaining Signal ($\\bar{\\alpha}_t$)", fontsize=12)

plt.title(
    "Comparison of Remaining Signal in Noise Schedules",
    fontsize=14,
    fontweight="bold"
)

plt.grid(alpha=0.3)

plt.legend()

plt.tight_layout()

plt.savefig(
    "remaining_signal_schedule.pdf",
    dpi=300,
    bbox_inches="tight"
)

plt.show()