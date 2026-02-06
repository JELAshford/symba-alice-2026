"""Run a system and quantify it's mutual info across the run"""

from symba_alice_2026.basic_barricelli import gather_replication_candidate, norm_zero
from scipy.spatial.distance import pdist, squareform
from pyinform.utils import coalesce_series
from pyinform.shannon import mutual_info
from pyinform.dist import Dist
from collections import Counter
from itertools import product
import matplotlib.pylab as plt
import numpy as np


def calc_entropy(world_a: np.ndarray, vocab: int = None) -> np.float64:
    """
    calculate Shannon entropy for a discrete array of CA states
    returns entropy in bits
    """

    if vocab is None:
        unique_states = np.unique(world_a)
    else:
        unique_states = vocab

    entropy = 0

    # discrete states
    entropy = 0.0
    for x_state in unique_states:
        px = (x_state == world_a).mean()
        if px:
            entropy += -px * np.log2(px)

    return entropy


def calc_mutual_info(world_a: np.ndarray, world_b: np.ndarray, base: float = 2.0):
    a_states, _ = coalesce_series(world_a)
    b_states, _ = coalesce_series(world_b)
    state_product = np.array(list(product(a_states, b_states)))
    combo_states = state_product[:, 0] * 2 + state_product[:, 1]

    p_x = Dist(a_states)
    p_y = Dist(b_states)
    p_xy = Dist(np.array(list(set(combo_states))))

    for a_val, b_val in zip(a_states, b_states):
        p_x.tick(a_val)
        p_y.tick(b_val)
        p_xy.tick(a_val * 2 + b_val)

    my_info = mutual_info(p_xy, p_x, p_y, b=base)

    return my_info


def calc_joint_entropy(traj: np.ndarray) -> float:
    """
    compute H(X_1, X_2, …, X_N) where each column of `traj`
    corresponds to a variable and each row is an observation.
    """
    # Treat each row as a joint outcome (tuple)
    joint_counts = Counter(map(tuple, traj))
    total = traj.shape[0]
    return -sum(
        (cnt / total) * np.log2(cnt / total) for cnt in joint_counts.values() if cnt > 0
    )


def transfer_entropy():
    pass


def active_information_storage():
    pass


if __name__ == "__main__":
    SIZE = 64
    TIMESTEPS = 128
    MAX_VAL = 2
    SEED = 1919264

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
    fig, ax = plt.subplots(1, 2, width_ratios=[0.7, 0.3], figsize=(7, 5))
    ax[0].imshow(grid, aspect="auto", interpolation="none")
    ax[0].tick_params(which="both", bottom=False, left=False, labelbottom=False)
    ax[0].set_xlabel("1D Space")
    ax[0].set_ylabel("Time")
    ax[1].plot(list(map(calc_entropy, grid)), np.arange(TIMESTEPS))
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Shannon Entropy")
    plt.savefig("out/entropy.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.show()

    # # Calcualte mutual info
    # mutual_info_comparison = squareform(pdist(grid, metric=calc_mutual_info))
    # np.fill_diagonal(
    #     mutual_info_comparison,
    #     list(
    #         map(
    #             lambda ind: calc_mutual_info(grid[ind, :], grid[ind, :]),
    #             np.arange(TIMESTEPS),
    #         ),
    #     ),
    # )

    # # Show mutual information
    # fig, ax = plt.subplots(1, 2, width_ratios=[0.7, 0.3], figsize=(10, 5))
    # ax[0].imshow(mutual_info_comparison)
    # ax[0].set_xlabel("Timestep")
    # ax[0].set_ylabel("Timestep")
    # ax[1].imshow(grid, aspect="auto", interpolation="none")
    # ax[1].tick_params(which="both", bottom=False, left=False, labelbottom=False)
    # ax[1].set_xlabel("1D Space")
    # ax[1].set_ylabel("Time")
    # plt.savefig("out/mutual_info.png", bbox_inches="tight", transparent=True, dpi=200)
    # plt.show()
