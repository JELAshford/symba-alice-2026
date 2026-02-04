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


# Show off each rule
TIMESTEPS = 512
MAX_VAL = 10
SIZE = 512
SEED = 8346
rng = np.random.default_rng(seed=SEED)
grid = np.zeros((TIMESTEPS, SIZE)).astype(int)
grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0

# Iteratively apply the replication updates/mutation norms
rule_sets = [norm_A, norm_B]
edges = np.linspace(0, SIZE, len(rule_sets) + 1).astype(int)
for step in range(1, TIMESTEPS):
    candidates = gather_replication_candidate(grid[step - 1, :])
    for rule, (start, end) in zip(rule_sets, zip(edges[:-1], edges[1:])):
        grid[step, start:end] = rule(grid[step - 1, start:end], candidates[start:end])

# View the final state!
fig, ax = plt.subplots(1, 1)
ax.imshow(grid, aspect="auto", interpolation="none", cmap="Blues")
plt.axis("off")
plt.savefig("out/combo.png", bbox_inches="tight", transparent=True, dpi=200)
plt.close()
