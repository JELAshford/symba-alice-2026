"""Create demos"""

from symba_alice_2026.basic_barricelli import (
    gather_replication_candidate,
    norm_zero,
    norm_A,
    norm_B,
    norm_C,
    norm_D,
)
import matplotlib.pylab as plt
import numpy as np

# Demo in paper (Fig 17. Numerical Testing of Evolution Theories)
TIMESTEPS = 30
SIZE = 60
grid = np.zeros((TIMESTEPS, SIZE)).astype(int)
grid[0, :] = (
    [0] * 20
    + [0, 0, 0, 0, 0, 1, -2, 1, 1, -2, 0, 1, -2, 1, 1, -2, 0, 0, 0, 0]
    + [0] * 20
)

# Iteratively apply the replication updates/mutation norms
for step in range(1, TIMESTEPS):
    candidates = gather_replication_candidate(grid[step - 1, :])
    grid[step, :] = norm_zero(grid[step - 1, :], candidates)
    grid[:, :20] = 0
    grid[:, -20:] = 0

# View the final state!
fig, ax = plt.subplots(1, 1)
ax.imshow(grid, aspect="auto", interpolation="none", cmap="gray")
plt.axis("off")
plt.savefig("out/fig17.png", bbox_inches="tight", transparent=True, dpi=200)
plt.close()


# Show off each rule
TIMESTEPS = 512
MAX_VAL = 10
SIZE = 512
SEED = 8346
rng = np.random.default_rng(seed=SEED)
grid = np.zeros((TIMESTEPS, SIZE)).astype(int)
grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0

rules = {"normA": norm_A, "normB": norm_B, "normC": norm_C, "normD": norm_D}
for name, rule in rules.items():
    this_grid = grid.copy()
    # Iteratively apply the replication updates/mutation norms
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(this_grid[step - 1, :])
        this_grid[step, :] = rule(this_grid[step - 1, :], candidates)

    # View the final state!
    fig, ax = plt.subplots(1, 1)
    ax.imshow(this_grid, aspect="auto", interpolation="none", cmap="Blues")
    plt.axis("off")
    plt.savefig(f"out/{name}.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.close()
