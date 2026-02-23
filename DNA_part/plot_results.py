import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict


# ── Load data ──
def load_metrics(path="metrics.csv"):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_survival(path="survival.csv"):
    with open(path) as f:
        return list(csv.DictReader(f))


# ── Organize by condition and cycle ──
def group_by_cond_cycle(rows, field, dtype=float):
    data = defaultdict(lambda: defaultdict(list))
    for r in rows:
        data[r["cond"]][int(r["cycle"])].append(dtype(r[field]))
    return data


def mean_std(data, cond):
    cycles = sorted(data[cond].keys())
    means = [np.mean(data[cond][c]) for c in cycles]
    stds = [np.std(data[cond][c]) for c in cycles]
    return np.array(cycles), np.array(means), np.array(stds)


# ── Colors and labels ──
COLORS = {
    "pair+elong+merge": "#2563eb",
    # "pair+elong":       "#dc2626",
    "elong_only":       "#16a34a",
}
LABELS = {
    "pair+elong+merge": "pairing + merging",
    # "pair+elong":       "pairing only",
    "elong_only":       "elongation only (null)",
}

def col(cond):
    return COLORS[cond]

def lab(cond):
    return LABELS[cond]


# ── Main plot ──
def plot_emergence(metrics_csv="metrics.csv", survival_csv="survival.csv",
                   output_png="emergence_results.png"):
    rows = load_metrics(metrics_csv)
    surv_rows = load_survival(survival_csv)

    conditions = sorted(set(r["cond"] for r in rows))

    # Data
    splits_data = group_by_cond_cycle(rows, "split_events")
    mean_len_data = group_by_cond_cycle(rows, "mean_len")

    # Prefer repeat-mass: fraction of k-mer windows that are repeats (non-singletons).
    # Fallback to legacy redundancy_k* if repeat_mass_fraction_k* is absent.
    kmer_metric = {}
    metric_label = None
    for k in range(2, 12):
        field = f"repeat_mass_fraction_k{k}"
        if field in rows[0]:
            kmer_metric[k] = group_by_cond_cycle(rows, field)
            metric_label = "k-mer repeat mass (%)"

    if len(kmer_metric) == 0:
        for k in range(2, 12):
            field = f"redundancy_k{k}"
            if field in rows[0]:
                kmer_metric[k] = group_by_cond_cycle(rows, field)
                metric_label = "k-mer redundancy (%)"

    if len(kmer_metric) == 0:
        raise RuntimeError("No repeat_mass_fraction_k* or redundancy_k* columns found in metrics.csv")

    surv_by_cond = defaultdict(lambda: defaultdict(list))
    for r in surv_rows:
        age = int(r["age"])
        if age <= 120:
            surv_by_cond[r["cond"]][age].append(float(r["frac_kmers_surviving"]))

    # ── Figure: 2 rows via gridspec ──
    ks = sorted(kmer_metric.keys())
    # if len(ks) == 0:
    #     raise RuntimeError("No k-mer metric columns found in metrics.csv")

    fig = plt.figure(figsize=(2.2 * len(ks) + 10, 10))
    outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.38)

    top = outer[0].subgridspec(1, len(ks), wspace=0.35)
    bottom = outer[1].subgridspec(1, 3, wspace=0.35)

    fig.suptitle(
        "Emergence signatures: does complementary association produce "
        "heritable sequence amplification?",
        fontsize=14, fontweight="bold", y=0.99,
    )

    # ── Row 1: k-mer redundancy panels (all ks present in CSV) ──
    for idx, k in enumerate(ks):
        ax = fig.add_subplot(top[0, idx])
        data = kmer_metric[k]
        for cond in conditions:
            if cond not in data or cond not in COLORS:  # Skip pair+elong
                continue
            c, m, s = mean_std(data, cond)
            ax.plot(c, m * 100, color=col(cond), linewidth=1.8,
                    label=lab(cond) if idx == 0 else None)
            ax.fill_between(c, (m - s) * 100, (m + s) * 100,
                            color=col(cond), alpha=0.10)
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel(metric_label, fontsize=8)
        ax.set_title(f"{k}-mer metric", fontsize=10, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=6, loc="upper left")

    # ── Row 2: 3 summary panels ──
    ax_splits = fig.add_subplot(bottom[0, 0])
    ax_length = fig.add_subplot(bottom[0, 1])
    ax_surv   = fig.add_subplot(bottom[0, 2])

    # ── Panel: Cumulative split events ──
    ax = ax_splits
    for cond in conditions:
        if cond not in splits_data or cond not in COLORS:  # Skip pair+elong
            continue
        c, m, s = mean_std(splits_data, cond)
        ax.plot(c, m, color=col(cond), linewidth=2.5, label=lab(cond))
        ax.fill_between(c, m - s, m + s, color=col(cond), alpha=0.12)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Cumulative split events")
    ax.set_title("Replication rate\n(split = ds → 2 ss children)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")

    # ── Panel: Mean piece length (pair+elong and pair+elong+merge only) ──
    ax = ax_length
    for cond in ["pair+elong+merge", "pair+elong"]:
        if cond not in mean_len_data or cond not in COLORS:  # Skip pair+elong
            continue
        c, m, s = mean_std(mean_len_data, cond)
        ax.plot(c, m, color=col(cond), linewidth=2, label=lab(cond))
        ax.fill_between(c, m - s, m + s, color=col(cond), alpha=0.12)
    ax.set_xlabel("Cycle")
    ax.set_ylabel("Mean piece length (columns)")
    ax.set_title("Piece growth\n(pairing conditions only)",
                 fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_ylim(bottom=0)

    # ── Panel: Child k-mer survival ──
    ax = ax_surv
    has_survival = False
    for cond in conditions:
        if cond not in surv_by_cond or not surv_by_cond[cond] or cond not in COLORS:  # Skip pair+elong
            continue
        ages = sorted(surv_by_cond[cond].keys())
        means = np.array([np.mean(surv_by_cond[cond][a]) for a in ages])
        stds = np.array([np.std(surv_by_cond[cond][a]) for a in ages])
        ages = np.array(ages)
        ax.plot(ages, means, color=col(cond), linewidth=2, label=lab(cond))
        ax.fill_between(ages, means - stds, means + stds,
                        color=col(cond), alpha=0.12)
        has_survival = True

    if has_survival:
        ax.set_xlabel("Cycles since split event")
        ax.set_ylabel("Fraction child k-mers surviving")
        ax.set_title("Inheritance persistence\n(do child motifs survive?)",
                     fontsize=10, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "No survival data\n(no splits occurred?)",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray")
        ax.set_title("Inheritance persistence", fontsize=10, fontweight="bold")

    # ── Footer ──
    n_seeds = len(set(r["seed"] for r in rows))
    max_cycle = max(int(r["cycle"]) for r in rows)
    fig.text(0.5, 0.01,
             f"{n_seeds} seeds × {max_cycle} cycles × {len(conditions)} conditions  |  "
             f"repeat mass = (total windows − singleton windows) / total windows  (fallback: 1 − unique/total)",
             ha="center", fontsize=8, color="gray")

    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    print(f"Saved {output_png}")




def plot_emergence_probability(metrics_csv="metrics.csv",
                              output_png="emergence_probability.png",
                              ks=None,
                              threshold=0.20,
                              alpha=0.05,
                              metric_prefix="repeat_mass_fraction_k"):
    if not isinstance(threshold, (float, int)):
        raise TypeError(f"threshold must be float or int, got {type(threshold)}")
    threshold = float(threshold)
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    if not isinstance(alpha, (float, int)):
        raise TypeError(f"alpha must be float or int, got {type(alpha)}")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    if not isinstance(metric_prefix, str):
        raise TypeError(f"metric_prefix must be str, got {type(metric_prefix)}")
    if len(metric_prefix) == 0:
        raise ValueError("metric_prefix must be non-empty")

    rows = load_metrics(metrics_csv)
    if len(rows) == 0:
        raise ValueError(f"No rows found in {metrics_csv}")

    # Determine available k values.
    header = set(rows[0].keys())
    if ks is None:
        inferred = []
        for k in range(2, 50):
            colname = f"{metric_prefix}{k}"
            if colname in header:
                inferred.append(k)
        if len(inferred) == 0:
            raise RuntimeError(
                f"No columns matching {metric_prefix}{{k}} found in metrics.csv"
            )
        ks = inferred
    else:
        if not isinstance(ks, (list, tuple)):
            raise TypeError(f"ks must be list/tuple of ints or None, got {type(ks)}")
        if len(ks) == 0:
            raise ValueError("ks must be non-empty when provided")
        for k in ks:
            if not isinstance(k, int):
                raise TypeError(f"All ks must be ints, got {type(k)}")
            if k < 1:
                raise ValueError(f"k must be >= 1, got {k}")
            colname = f"{metric_prefix}{k}"
            if colname not in header:
                raise KeyError(f"Column not found in metrics.csv: {colname}")

    conditions = sorted(set(r["cond"] for r in rows))

    # --- Binomial CI: Wilson score interval ---
    def _wilson_interval(k_success, n_total, alpha_level):
        if not isinstance(k_success, (int, np.integer)):
            raise TypeError(f"k_success must be int, got {type(k_success)}")
        if not isinstance(n_total, (int, np.integer)):
            raise TypeError(f"n_total must be int, got {type(n_total)}")
        if n_total <= 0:
            raise ValueError(f"n_total must be > 0, got {n_total}")
        if k_success < 0 or k_success > n_total:
            raise ValueError(f"k_success must be in [0, n_total], got {k_success}")
        if not (0.0 < alpha_level < 1.0):
            raise ValueError(f"alpha_level must be in (0, 1), got {alpha_level}")

        import scipy.stats as stats

        z = float(stats.norm.ppf(1.0 - alpha_level / 2.0))
        phat = float(k_success) / float(n_total)
        denom = 1.0 + (z * z) / float(n_total)
        center = (phat + (z * z) / (2.0 * float(n_total))) / denom
        rad = (z / denom) * np.sqrt(
            (phat * (1.0 - phat) / float(n_total)) + (z * z) / (4.0 * float(n_total) ** 2)
        )
        lo = center - rad
        hi = center + rad
        if lo < 0.0:
            lo = 0.0
        if hi > 1.0:
            hi = 1.0
        return float(lo), float(hi)

    # --- Plot ---
    fig = plt.figure(figsize=(2.1 * len(ks) + 3.5, 4.2))
    gs = gridspec.GridSpec(1, len(ks), figure=fig, wspace=0.35)

    for idx, k in enumerate(ks):
        ax = fig.add_subplot(gs[0, idx])
        field = f"{metric_prefix}{k}"
        data = group_by_cond_cycle(rows, field)

        for cond in conditions:
            if cond not in data:
                continue
            if cond not in COLORS:
                continue
            cycles = sorted(data[cond].keys())
            fracs = []
            los = []
            his = []
            for c in cycles:
                vals = np.asarray(data[cond][c], dtype=float)
                if vals.ndim != 1:
                    raise ValueError(
                        f"Expected 1D values for cond={cond}, cycle={c}, got shape {vals.shape}"
                    )
                if len(vals) == 0:
                    raise ValueError(f"Empty list for cond={cond}, cycle={c}")
                k_success = int(np.sum(vals >= threshold))
                n_total = int(len(vals))
                p = float(k_success) / float(n_total)
                lo, hi = _wilson_interval(k_success, n_total, alpha)
                fracs.append(p)
                los.append(lo)
                his.append(hi)

            cycles_arr = np.asarray(cycles, dtype=int)
            fracs_arr = np.asarray(fracs, dtype=float)
            los_arr = np.asarray(los, dtype=float)
            his_arr = np.asarray(his, dtype=float)

            ax.plot(
                cycles_arr,
                fracs_arr,
                color=col(cond),
                linewidth=1.8,
                label=lab(cond) if idx == 0 else None,
            )
            ax.fill_between(
                cycles_arr,
                los_arr,
                his_arr,
                color=col(cond),
                alpha=0.15,
            )

        ax.set_title(f"{k}-mer", fontsize=10, fontweight="bold")
        ax.set_xlabel("Cycle", fontsize=8)
        ax.set_ylabel("P(emerged)", fontsize=8)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc="lower right")

    thr_pct = 100.0 * threshold
    fig.suptitle(
        f"Emergence probability (metric ≥ {thr_pct:.1f}%) with {int((1.0-alpha)*100)}% Wilson CI",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    print(f"Saved {output_png}")



if __name__ == "__main__":
    plot_emergence()
    plot_emergence_probability(
        metrics_csv="metrics.csv",
        output_png="emergence_prob_k2to8_thr10.png",
        ks=[2,3,4,5,6,7,8],
        threshold=0.10,  # 10% repeat-mass threshold
        alpha=0.05
    )