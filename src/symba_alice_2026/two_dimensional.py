"""2d Barricelli"""

from matplotlib.animation import ArtistAnimation
import matplotlib.pylab as plt
from itertools import product
import numpy as np

SIZE = 128
TIMESTEPS = 1024
MAX_VAL = 3
SEED = 1701


def tuple_add(t1, t2):
    return tuple([v1 + v2 for v1, v2 in zip(t1, t2)])


def tuple_sub(t1, t2):
    return tuple([v1 - v2 for v1, v2 in zip(t1, t2)])


def tuple_mul(t1, v: int):
    return tuple([v1 * v for v1 in t1])


def tuple_mod(t1, t2):
    return tuple(v1 % v2 for v1, v2 in zip(t1, t2))


def gather_replication_candidate(state: np.ndarray) -> list[set]:
    h, w = len(state), len(state[0])
    coords = product(range(h), range(w))
    candidates = [[set() for _ in range(w)] for _ in range(h)]
    for y, x in coords:
        current_val = state[y][x]
        pos = (y, x)
        # skip if current value is zero as nothing happens
        if current_val == (0, 0):
            continue
        # gather all the offsets that this current_value will be a candidate for
        offsets = {current_val}
        offset_queue = [current_val]
        while offset_queue:
            check_offset = offset_queue.pop()
            new_pos = tuple_mod(tuple_add(pos, check_offset), (h, w))
            value_at_offset = state[new_pos[0]][new_pos[1]]
            if value_at_offset not in offsets:
                offsets.add(value_at_offset)
                offset_queue.append(value_at_offset)
            else:
                break
        # add as a candidate at all those offsets
        for offset in offsets:
            new_pos = tuple_mod(tuple_add(pos, offset), (h, w))
            if offset != (0, 0):
                candidates[new_pos[0]][new_pos[1]].add(current_val)
    return candidates


def norm_D(
    current_state: np.ndarray,
    collisions: list[set],
):
    # If no collision, return the value that got written to that position
    h, w = len(current_state), len(current_state[0])
    coords = product(range(h), range(w))
    new_state = [[(0, 0)] * w for _ in range(h)]
    for y, x in coords:
        candidates = collisions[y][x]
        current_val = current_state[y][x]
        if len(candidates) == 1:
            new_state[y][x] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != (0, 0):
            continue
        else:
            # There is a collision
            prev_pos = tuple_mod(tuple_sub((y, x), current_val), (h, w))
            next_pos = tuple_mod(tuple_add((y, x), current_val), (h, w))
            prev_val = current_state[prev_pos[0]][prev_pos[1]]
            next_val = current_state[next_pos[0]][next_pos[1]]
            if prev_val == next_val:
                new_state[y][x] = tuple_sub(tuple_mul(next_val, 2), current_val)
            else:
                continue
    return new_state


def norm_zero(
    current_state: np.ndarray,
    collisions: list[set],
):
    h, w = len(current_state), len(current_state[0])
    coords = product(range(h), range(w))
    new_state = [[(0, 0)] * w for _ in range(h)]
    for y, x in coords:
        these_collisions = collisions[y][x]
        if len(these_collisions) == 1:
            new_state[y][x] = list(these_collisions)[0]
    return new_state


if __name__ == "__main__":
    # Basic baricelli copying
    rng = np.random.default_rng(seed=SEED)
    grid = np.zeros((TIMESTEPS, SIZE, SIZE, 2)).astype(int)
    grid[0, ...] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE, SIZE, 2))
    grid[0, ...] *= (rng.random((SIZE, SIZE, 2)) < 0.8).astype(int)
    grid = [
        [[tuple(map(int, s)) for s in row] for row in timestep] for timestep in grid
    ]

    # Iteratively apply the replication updates/mutation norms
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(grid[step - 1])
        grid[step] = norm_zero(grid[step - 1], candidates)

    # # View the final state!
    # fig, ax = plt.subplots(1, 1)
    # vis_grid = np.array(grid)[0, ...].mean(axis=2)
    # ax.imshow(vis_grid, aspect="auto", interpolation="none")
    # plt.axis("off")
    # plt.savefig("out/basic.png", bbox_inches="tight", transparent=True, dpi=200)
    # plt.show()

    # View the full run
    frames = np.array(grid).mean(axis=3)

    fig, ax = plt.subplots(1, 1)
    artists = []
    for frame in frames[::2]:
        display = ax.imshow(frame, vmin=frames.min(), vmax=frames.max())
        artists.append([display])

    ani = ArtistAnimation(fig=fig, artists=artists, interval=50)
    ani.save("out/2d.mp4")
    plt.close()

    # Do a transplant
    replicator_sample = np.array(grid)[-1, 40:60, 70:90, :]
    new_grid = np.zeros((512, SIZE, SIZE, 2)).astype(int)
    new_grid[0, 40:60, 40:60, :] = replicator_sample
    new_grid = [
        [[tuple(map(int, s)) for s in row] for row in timestep] for timestep in new_grid
    ]

    for step in range(1, 512):
        candidates = gather_replication_candidate(new_grid[step - 1])
        new_grid[step] = norm_zero(new_grid[step - 1], candidates)

    # View the transplant run
    transplant_frames = np.array(new_grid).mean(axis=3)

    fig, ax = plt.subplots(1, 1)
    artists = []
    for frame in transplant_frames[::2]:
        display = ax.imshow(
            frame,
            vmin=transplant_frames.min(),
            vmax=transplant_frames.max(),
        )
        artists.append([display])

    ani_transplant = ArtistAnimation(fig=fig, artists=artists, interval=50)
    ani_transplant.save("out/2d_transplant.mp4")
    plt.close()

    # Random patch
    new_grid = np.zeros((512, SIZE, SIZE, 2)).astype(int)
    new_grid[0, 40:60, 40:60, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(20, 20, 2))
    new_grid = [
        [[tuple(map(int, s)) for s in row] for row in timestep] for timestep in new_grid
    ]

    for step in range(1, 512):
        candidates = gather_replication_candidate(new_grid[step - 1])
        new_grid[step] = norm_zero(new_grid[step - 1], candidates)

    patch_frames = np.array(new_grid).mean(axis=3)

    fig, ax = plt.subplots(1, 1)
    artists = []
    for frame in patch_frames[::2]:
        display = ax.imshow(
            frame,
            vmin=patch_frames.min(),
            vmax=patch_frames.max(),
        )
        artists.append([display])

    ani_patch = ArtistAnimation(fig=fig, artists=artists, interval=50)
    ani_patch.save("out/2d_patch.mp4")
    plt.close()
