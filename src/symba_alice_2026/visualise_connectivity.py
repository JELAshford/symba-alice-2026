"""Create a visialisation of connectivity between positions in the barricelli strings"""

from symba_alice_2026.basic_barricelli import gather_replication_candidate, norm_zero
from matplotlib.patches import ConnectionPatch
from matplotlib.gridspec import GridSpec
import matplotlib.pylab as plt
import networkx as nx
import numpy as np

SIZE = 512
SEED = 198264
TIMESTEPS = 256
MAX_VAL = 10


def get_state_connections(state):
    size = len(state)
    connections = [set() for _ in range(size)]
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
                connections[new_pos].add(pos)
    return connections


def get_integer_connections(state):
    """Create a graph of all"""


if __name__ == "__main__":
    # Basic baricelli copying
    rng = np.random.default_rng(seed=SEED)
    grid = np.zeros((TIMESTEPS, SIZE)).astype(int)

    # Initialise with sparse random
    grid[0, :] = rng.integers(-MAX_VAL, MAX_VAL + 1, size=(SIZE))
    # grid[0, rng.choice(np.arange(SIZE), size=int(SIZE * (4 / 5)), replace=False)] = 0
    # grid[0, :] = np.zeros(SIZE)
    # grid[0, 100:102] = [1, -1]

    # Iteratively apply the replication updates/mutation norms
    for step in range(1, TIMESTEPS):
        candidates = gather_replication_candidate(grid[step - 1, :])
        grid[step, :] = norm_zero(grid[step - 1, :], candidates)

    # View the final state!
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grid, aspect="auto", interpolation="none")
    plt.axis("off")
    plt.savefig("out/vis.png", bbox_inches="tight", transparent=True, dpi=200)
    plt.show()

    # Show them all on subplots
    num_graphs = 10

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(num_graphs, 2, figure=fig, width_ratios=[0.8, 0.2])
    image_ax = fig.add_subplot(gs[:, 0])
    image_ax.imshow(grid, aspect="auto", interpolation="none")
    image_ax.set_axis_off()

    # Calcualte the connectivity in the grid state - store links between positions and where they "jump" to
    for ind, sample_ind in enumerate(
        np.linspace(0, TIMESTEPS - 1, num_graphs).astype(int)
    ):
        connections = get_state_connections(grid[sample_ind, :])
        edges = [(ind, conn) for ind, conns in enumerate(connections) for conn in conns]
        # builld graph
        graph = nx.DiGraph()
        graph.add_edges_from(edges)
        # draw graph
        options = {
            "font_size": 12,
            "node_size": 2,
            "node_color": "white",
            "edgecolors": "black",
            "linewidths": 1,
            "width": 1,
        }
        view_ax = fig.add_subplot(gs[ind, 1])
        pos = nx.spring_layout(graph, method="energy")
        nx.draw(graph, **options, ax=view_ax)
        # add arrow to correct row
        all_positions = np.stack([v for k, v in pos.items()])
        con = ConnectionPatch(
            xyA=[all_positions[:, 0].min(), all_positions[:, 1].mean()],
            coordsA=view_ax.transData,
            xyB=[SIZE, sample_ind],
            coordsB=image_ax.transData,
            color="black",
            arrowstyle="-|>",
            mutation_scale=30,  # controls arrow head size
            linewidth=3,
        )
        fig.add_artist(con)

    plt.tight_layout()
    plt.show()
