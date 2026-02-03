from collections import Counter
import matplotlib.pylab as plt
import numpy as np

SIZE = 256
TIMESTEPS = 100
MAX_VAL = 10
SEED = 1701

# Basic baricelli copying
rng = np.random.default_rng()
grid = np.zeros((TIMESTEPS, SIZE)).astype(int)

# Initialise with random
grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))

# Loop
for step in range(1, TIMESTEPS):
    # Move integers, handling collisions (mutations) with addition!
    for pos, val in enumerate(grid[step - 1, :]):
        grid[step, (pos + val) % SIZE] += val

# What's the composition of the final state?
print(Counter(grid[-1, :]))

# View the final state!
fig, ax = plt.subplots(1, 1)
ax.imshow(grid / MAX_VAL, aspect="auto", interpolation="none", cmap="Blues")
plt.savefig("out.png", bbox_inches="tight", transparent=True, dpi=200)
plt.show()
