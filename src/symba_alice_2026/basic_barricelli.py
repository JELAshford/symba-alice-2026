import matplotlib.pylab as plt
import numpy as np

SIZE = 512
TIMESTEPS = 128
MAX_VAL = 10
SEED = 1701


def gather_replication_candidate(state: np.ndarray) -> list[set]:
    size = len(state)
    candidates = [set() for _ in range(size)]
    for pos, current_val in enumerate(state):
        # skip if current value is zero as nothing happens
        if current_val == 0:
            continue
        # gather all the offsets that this current_value will be a candidate for
        offsets = {current_val}
        offset_queue = [current_val]
        while offset_queue:
            check_offset = offset_queue.pop()
            value_at_offset = state[(pos + check_offset) % size]
            if value_at_offset not in offsets:
                offsets.add(value_at_offset)
                offset_queue.append(value_at_offset)
            else:
                break
        # add as a candidate at all those offsets
        for offset in offsets:
            new_pos = (pos + offset) % size
            if offset != 0:
                candidates[new_pos].add(int(current_val))
    return candidates


def norm_zero(
    current_state: np.ndarray,
    collisions: list[set],
) -> int:
    return np.array([list(vals)[0] if len(vals) == 1 else 0 for vals in collisions])


def norm_A(
    current_state: np.ndarray,
    collisions: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(collisions, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != 0:
            continue
        else:
            # There is a collision
            # search left for nearest active value
            left_dist = 1
            left_pos = (ind - 1) % size
            while (left_val := current_state[left_pos]) == 0:
                left_pos = (left_pos - 1) % size
                left_dist += 1

            right_dist = 1
            right_pos = (ind + 1) % size
            while (right_val := current_state[right_pos]) == 0:
                right_pos = (right_pos + 1) % size
                right_dist += 1

            if np.sign(left_val) == np.sign(right_val):
                new_state[ind] = left_dist + right_dist
            else:
                new_state[ind] = -(left_dist + right_dist)
    return new_state


def norm_B(
    current_state: np.ndarray,
    collisions: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(collisions, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != 0:
            continue
        else:
            # There is a collision
            # search left for nearest active value
            left_dist = 1
            left_pos = (ind - 1) % size
            while (left_val := current_state[left_pos]) == 0:
                left_pos = (left_pos - 1) % size
                left_dist += 1

            right_dist = 1
            right_pos = (ind + 1) % size
            while (right_val := current_state[right_pos]) == 0:
                right_pos = (right_pos + 1) % size
                right_dist += 1

            if np.sign(left_val) == np.sign(right_val):
                new_state[ind] = left_dist + right_dist - 1
            else:
                new_state[ind] = -(left_dist + right_dist - 1)
    return new_state


def norm_C(
    current_state: np.ndarray,
    collisions: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(collisions, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        else:
            # There is a collision
            # search left for nearest active value
            left_pos = (ind - 1) % size
            while (left_val := current_state[left_pos]) == 0:
                left_pos = (left_pos - 1) % size

            right_pos = (ind + 1) % size
            while (right_val := current_state[right_pos]) == 0:
                right_pos = (right_pos + 1) % size

            new_state[ind] = left_val - right_val
    return new_state


def norm_D(
    current_state: np.ndarray,
    collisions: list[set],
):
    # If no collision, return the value that got written to that position
    size = len(current_state)
    new_state = np.zeros_like(current_state)
    for ind, (candidates, current_val) in enumerate(zip(collisions, current_state)):
        if len(candidates) == 1:
            new_state[ind] = list(candidates)[0]
        elif len(candidates) == 0:
            continue
        elif current_val != 0:
            continue
        else:
            # There is a collision
            left_val = current_state[(ind - current_val) % size]
            right_val = current_state[(ind + current_val) % size]
            if left_val == right_val:
                new_state[ind] = -current_val + 2 * right_val
            else:
                continue
    return new_state


if __name__ == "__main__":
    # Basic baricelli copying
    rng = np.random.default_rng(seed=SEED)
    grid = np.zeros((TIMESTEPS, SIZE)).astype(int)

    # Initialise with sparse random
    grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
    grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0

    # Iteratively apply the replication updates/mutation norms
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(grid[step - 1, :])
        grid[step, :] = norm_zero(grid[step - 1, :], candidates)

    # View the final state!
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grid, aspect="auto", interpolation="none")
    plt.axis("off")
    plt.savefig("out/basic.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.show()
