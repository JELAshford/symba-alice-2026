import random
import csv  # to write CSV outputs
import os   # to create directories for frame PNGs
from collections import Counter, defaultdict    # for k-mer statistics (Counter) and survival tracking (defaultdict)
import matplotlib.pyplot as plt
import sys
from contextlib import redirect_stdout, redirect_stderr

# ================================================
# CONSTANTS
# ================================================
NUCLEOTIDES = ["A", "T", "C", "G"]
COMP = {"A": "T", "T": "A", "C": "G", "G": "C"} 

MIN_DS_FOR_SPLIT = 8         
MU = 0.0001                   
KMER_RANGE = range(2, 12)     


# ================================================
# ID GENERATOR
# ================================================
NEXT_ID = 1     

def new_id():
    global NEXT_ID    
    i = NEXT_ID    
    NEXT_ID += 1      
    return i        


# ================================================
# PIECE CLASS
# ================================================
class Piece:
    def __init__(self, cols=None, generation=0, parent_id=None,
                 birth_cycle=0, mu=MU, pid=None,
                 sibling_id=None, can_elongate=True):
        self.id = new_id() if pid is None else pid
        self.cols = cols[:] if cols else []
        self.generation = generation
        self.parent_id = parent_id
        self.birth_cycle = birth_cycle
        self.mu = mu
        self.sibling_id = sibling_id
        self.can_elongate = can_elongate

    def __len__(self):
        return len(self.cols)

    def is_empty(self):
        return len(self.cols) == 0

    def is_complete(self):
        if not self.cols:
            return False
        for t, b, _ in self.cols:
            if t is None or b is None or COMP[t] != b:
                return False
        return True

    def has_top(self):
        return any(t is not None for t, _, _ in self.cols)

    def has_bottom(self):
        return any(b is not None for _, b, _ in self.cols)

    def ancestry(self):
        s = set()
        for _, _, origin in self.cols:
            s |= origin
        return s

    def one_line(self):
        if not self.cols:
            return "[empty]"
        parts = []
        for t, b, _ in self.cols:
            if t is not None and b is not None:
                if t not in COMP:
                    raise ValueError(f"Invalid nucleotide in top strand: {t}")
                if COMP[t] == b:
                    parts.append(f"{t}-{b}")
                else:
                    parts.append(f"{t}{b}")
            elif t is not None:
                parts.append(f"{t}_")
            elif b is not None:
                parts.append(f"_{b}")
            else:
                parts.append("__")
        return " ".join(parts)

    @staticmethod
    def single(nuc, birth_cycle=0):
        p = Piece([], generation=0, birth_cycle=birth_cycle)
        p.cols = [(nuc, None, {p.id})]
        return p


# ================================================
# SEQUENCE HELPERS
# ================================================
def top_runs(p):
    runs = []
    current = []
    for t, _, _ in p.cols:
        if t is not None:
            current.append(t)
        else:
            if current:
                runs.append("".join(current))
                current = []
    if current:
        runs.append("".join(current))
    return runs

def bottom_runs(p):
    runs = []
    current = []
    for _, b, _ in p.cols:
        if b is not None:
            current.append(b)
        else:
            if current:
                runs.append("".join(current))
                current = []
    if current:
        runs.append("".join(current))
    return runs

def extract_kmers(seq, k):
    if len(seq) < k:
        return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]

def soup_kmer_set(soup, k):
    kmers = set()
    for p in soup:
        for run in top_runs(p):
            for km in extract_kmers(run, k):
                kmers.add(km)
        for run in bottom_runs(p):
            for km in extract_kmers(run, k):
                kmers.add(km)
    return kmers

def all_strand_sequences(soup):
    seqs = []
    for p in soup:
        seqs.extend(top_runs(p))
        seqs.extend(bottom_runs(p))
    return seqs


# ================================================
# COLUMN PREDICATES
# ================================================
def _is_paired(t, b):
    return t is not None and b is not None and COMP[t] == b

def _is_mismatch(t, b):
    return t is not None and b is not None and COMP[t] != b


# ================================================
# MUTATION
# ================================================
def mutate_soup(soup, mu=MU):
    for p in soup:
        new_cols = []
        for t, b, origin in p.cols:
            if t is not None and random.random() < mu:
                t = random.choice([x for x in NUCLEOTIDES if x != t])
            if b is not None and random.random() < mu:
                b = random.choice([x for x in NUCLEOTIDES if x != b])
            new_cols.append((t, b, origin))
        p.cols = new_cols


# ================================================
# MISMATCH-INDUCED SPLITTING 
# ================================================
def mismatch_split(piece, cycle):
    if len(piece.cols) == 0:
        return [piece]

    has_mm = any(_is_mismatch(t, b) for t, b, _ in piece.cols)
    if not has_mm:
        return [piece]

    segments = []
    current = []
    for t, b, origin in piece.cols:
        if _is_mismatch(t, b):
            if current:
                segments.append(current)
                current = []
            # Emit the two mismatched nucleotides as separate ss pieces
            segments.append([(t, None, origin.copy())])
            segments.append([(None, b, origin.copy())])
        else:
            current.append((t, b, origin))
    if current:
        segments.append(current)

    result = []
    for seg_cols in segments:
        if not seg_cols:
            continue
        p = Piece(seg_cols, generation=piece.generation,
                  parent_id=piece.id, birth_cycle=cycle,
                  mu=piece.mu, can_elongate=piece.can_elongate)
        result.append(p)
    return result


def mismatch_split_soup(soup, cycle):
    new_soup = []
    for p in soup:
        new_soup.extend(mismatch_split(p, cycle))
    soup[:] = new_soup


# ================================================
# PHASE 1: NUCLEOTIDE ABSORPTION (size-weighted)
# ================================================
def attach_or_pair(piece, nuc, allow_pairing=True, verbose=False):
    p_pair = 0.5
    p_long = 0.25
    p_alone = 0.25

    actions = []
    weights = []

    if allow_pairing:
        actions.append("pair")
        weights.append(p_pair)
    if piece.can_elongate:
        actions.append("longitudinal")
        weights.append(p_long)
    actions.append("alone")
    weights.append(p_alone)

    action = random.choices(actions, weights=weights)[0]

    if action == "alone":
        return piece, False

    # ── Pairing: fill a complementary gap ──
    if action == "pair":
        cols = piece.cols[:]

        for i, (t, b, origin) in enumerate(cols):
            if t is not None and b is None and COMP[t] == nuc:
                cols[i] = (t, nuc, origin)
                if verbose:
                    print(f"    nuc {nuc}: pairs into gap at col {i} with top {t}")
                return Piece(cols, pid=piece.id, generation=piece.generation,
                             parent_id=piece.parent_id, birth_cycle=piece.birth_cycle,
                             mu=piece.mu, sibling_id=piece.sibling_id,
                             can_elongate=piece.can_elongate), True

        for i, (t, b, origin) in enumerate(cols):
            if b is not None and t is None and COMP[nuc] == b:
                cols[i] = (nuc, b, origin)
                if verbose:
                    print(f"    nuc {nuc}: pairs into gap at col {i} with bottom {b}")
                return Piece(cols, pid=piece.id, generation=piece.generation,
                             parent_id=piece.parent_id, birth_cycle=piece.birth_cycle,
                             mu=piece.mu, sibling_id=piece.sibling_id,
                             can_elongate=piece.can_elongate), True

        if verbose:
            print(f"    nuc {nuc}: wanted to pair but found no compatible gap")
        return piece, False

    # ── Longitudinal: extend one strand end ──
    if action == "longitudinal":
        cols = piece.cols[:]
        piece_origin = piece.ancestry()
        has_top = piece.has_top()
        has_bot = piece.has_bottom()

        if has_top and has_bot:
            strand = random.choice(["top", "bottom"])
        elif has_top:
            strand = "top"
        elif has_bot:
            strand = "bottom"
        else:
            strand = "top"

        if strand == "top":
            cols.append((nuc, None, piece_origin.copy()))
            if verbose:
                print(f"    nuc {nuc}: extends TOP to the right")
        else:
            cols.insert(0, (None, nuc, piece_origin.copy()))
            if verbose:
                print(f"    nuc {nuc}: extends BOTTOM to the left")

        return Piece(cols, pid=piece.id, generation=piece.generation,
                     parent_id=piece.parent_id, birth_cycle=piece.birth_cycle,
                     mu=piece.mu, sibling_id=piece.sibling_id,
                     can_elongate=piece.can_elongate), True

    return piece, False


def absorb_nucleotide(soup, nuc, allow_pairing, cycle, verbose=False):
    if not soup:
        soup.append(Piece.single(nuc, birth_cycle=cycle))
        if verbose:
            print(f"  Cycle {cycle}: nuc {nuc} → new piece (empty soup)")
        return

    candidates = list(range(len(soup)))
    consumed = False

    while candidates and not consumed:
        wts = [max(len(soup[i]), 1) for i in candidates]
        pick = random.choices(range(len(candidates)), weights=wts)[0]
        idx = candidates[pick]

        soup[idx], consumed = attach_or_pair(
            soup[idx], nuc, allow_pairing=allow_pairing, verbose=verbose
        )
        if not consumed:
            candidates.pop(pick)

    if not consumed:
        soup.append(Piece.single(nuc, birth_cycle=cycle))
        if verbose:
            print(f"  Cycle {cycle}: nuc {nuc} → new piece (no taker)")


# ================================================
# PHASE 2: SPLITTING (replication)
# ================================================
def split(piece, cycle, verbose=False):
    cols = piece.cols

    if len(cols) < MIN_DS_FOR_SPLIT:
        if verbose:
            print(f"  split blocked: len={len(cols)} < {MIN_DS_FOR_SPLIT}: {piece.one_line()}")
        return [piece]

    child_gen = piece.generation + 1

    top_child = Piece(
        [(t, None, origin.copy()) for (t, b, origin) in cols],
        generation=child_gen, parent_id=piece.id, birth_cycle=cycle,
        mu=piece.mu, can_elongate=False,
    )
    bot_child = Piece(
        [(None, b, origin.copy()) for (t, b, origin) in cols],
        generation=child_gen, parent_id=piece.id, birth_cycle=cycle,
        mu=piece.mu, can_elongate=False,
    )
    top_child.sibling_id = bot_child.id
    bot_child.sibling_id = top_child.id

    if verbose:
        print(f"  split: {piece.one_line()} → top: {top_child.one_line()} | bot: {bot_child.one_line()}")

    return [top_child, bot_child]


# ================================================
# PHASE 3: MERGING (complementary association)
# ================================================
def _score_cols(cols):
    s = 0
    for t, b, _ in cols:
        if t is not None:
            s += 1
        if b is not None:
            s += 1
        if _is_paired(t, b):
            s += 1
    return s


def _has_internal_holes(cols):
    paired_idx = [
        i for i, (t, b, _) in enumerate(cols)
        if _is_paired(t, b)
    ]
    if len(paired_idx) < 2:
        return False
    L, R = paired_idx[0], paired_idx[-1]
    for i in range(L, R + 1):
        t, b, _ = cols[i]
        if not _is_paired(t, b):
            return True
    return False


def _combine_col(c1, c2):
    t1, b1, o1 = c1
    t2, b2, o2 = c2

    # Reject overlap at already-paired columns
    if _is_paired(t1, b1):
        return None, False
    if _is_paired(t2, b2):
        return None, False

    # Reject conflicting non-None values
    if t1 is not None and t2 is not None and t1 != t2:
        return None, False
    if b1 is not None and b2 is not None and b1 != b2:
        return None, False

    t = t1 if t1 is not None else t2
    b = b1 if b1 is not None else b2

    if t is not None and b is not None and COMP[t] != b:
        return None, False

    return (t, b, o1 | o2), True


def try_merge(a, b, verbose=False):
    if a.is_empty() or b.is_empty():
        return None, 0

    max_k = min(len(a), len(b))
    before_score = _score_cols(a.cols) + _score_cols(b.cols)

    for k in range(max_k, 0, -1):
        merged_cols = list(a.cols[:len(a.cols) - k])

        ok = True
        for i in range(k):
            cA = a.cols[len(a.cols) - k + i]
            cB = b.cols[i]
            combined, success = _combine_col(cA, cB)
            if not success:
                ok = False
                break
            merged_cols.append(combined)

        if not ok:
            continue

        merged_cols.extend(b.cols[k:])

        if _has_internal_holes(merged_cols):
            continue

        after_score = _score_cols(merged_cols)
        if after_score <= before_score:
            continue

        merged_piece = Piece(merged_cols, mu=a.mu)
        if verbose:
            print(f"  merge (overlap={k}): {a.one_line()} + {b.one_line()} → {merged_piece.one_line()}")
        return merged_piece, k

    return None, 0


def do_merging_phase(soup, verbose=False):
    changed = True
    while changed:
        changed = False

        candidates = []
        for i in range(len(soup)):
            for j in range(i + 1, len(soup)):
                if soup[i].sibling_id is not None and soup[i].sibling_id == soup[j].id:
                    continue
                if soup[j].sibling_id is not None and soup[j].sibling_id == soup[i].id:
                    continue

                merged, overlap_k = try_merge(soup[i], soup[j], verbose=False)
                if merged is None:
                    merged, overlap_k = try_merge(soup[j], soup[i], verbose=False)
                if merged is not None:
                    candidates.append((overlap_k, len(merged), i, j, merged))

        if not candidates:
            break

        candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)

        best_overlap, best_len, i, j, merged = candidates[0]
        if verbose:
            print(f"  merge applied (overlap={best_overlap}, len={best_len}): "
                  f"{soup[i].one_line()} + {soup[j].one_line()} → {merged.one_line()}")

        hi, lo = max(i, j), min(i, j)
        soup.pop(hi)
        soup.pop(lo)
        soup.append(merged)
        soup[:] = [p for p in soup if not p.is_empty()]
        changed = True

    return soup


# ================================================
# CARRYING CAPACITY
# ================================================
def degrade_soup(soup, capacity):
    if len(soup) <= capacity:
        return soup
    soup.sort(key=lambda p: (p.is_complete(), len(p)), reverse=True)
    return soup[:capacity]


# ================================================
# SPLIT TRACKER (inheritance persistence measurement)
# ================================================
class SplitTracker:
    def __init__(self, k=3):
        self.k = k
        self.events = []
        self.survival = defaultdict(list)

    def log_split(self, cycle, parent_piece, child_top, child_bot):
        child_top_seq = top_runs(child_top)
        child_bot_seq = bottom_runs(child_bot)

        child_kmers = set()
        for run in child_top_seq:
            for km in extract_kmers(run, self.k):
                child_kmers.add(km)
        for run in child_bot_seq:
            for km in extract_kmers(run, self.k):
                child_kmers.add(km)

        
        self.events.append({
            "event_idx": len(self.events),
            "cycle": cycle,
            "parent_len": len(parent_piece),
            "child_top_seq": child_top_seq,
            "child_bot_seq": child_bot_seq,
            "child_kmers": child_kmers,
        })

    def scan_soup(self, soup, current_cycle, max_track_age=150):
        s_kmers = soup_kmer_set(soup, self.k)

        for ev in self.events:
            age = current_cycle - ev["cycle"]
            if age < 0 or age > max_track_age:
                continue
            if not ev["child_kmers"]:
                continue

            surviving = ev["child_kmers"] & s_kmers
            frac = len(surviving) / len(ev["child_kmers"])

            self.survival[ev["event_idx"]].append({
                "age": age,
                "cycle": current_cycle,
                "frac_kmers_surviving": frac,
                "n_surviving": len(surviving),
                "total_child_kmers": len(ev["child_kmers"]),
            })

    def compute_survival_rows(self):
        rows = []
        for ev in self.events:
            eidx = ev["event_idx"]
            if eidx in self.survival:
                snaps = self.survival[eidx]
            else:
                snaps = []
            for snap in snaps:
                rows.append({
                    "event_idx": eidx,
                    "split_cycle": ev["cycle"],
                    "parent_len": ev["parent_len"],
                    "n_child_kmers": len(ev["child_kmers"]),
                    **snap,
                })
        return rows


# ================================================
# PER-CYCLE METRICS
# ================================================
def compute_metrics(soup, cycle, seed, cond, split_events_total, k=3):
    n_pieces = len(soup)
    n_ds = sum(1 for p in soup if p.is_complete())
    lengths = [len(p) for p in soup]
    mean_len = sum(lengths) / len(lengths) if lengths else 0.0
    max_len = max(lengths) if lengths else 0
    generations = [p.generation for p in soup]

    strand_seqs = all_strand_sequences(soup)
    total_strands = len(strand_seqs)
    seq_counter = Counter(strand_seqs)
    unique_seqs = len(seq_counter)
    top1_count = seq_counter.most_common(1)[0][1] if seq_counter else 0
    redundancy = 1.0 - (unique_seqs / total_strands) if total_strands > 0 else 0.0

    kmer_set = soup_kmer_set(soup, k)
    total_possible = 4 ** k
    occupancy = len(kmer_set) / total_possible

    result = {
        "seed": seed,
        "cycle": cycle,
        "cond": cond,
        "pieces": n_pieces,
        "n_ds": n_ds,
        "mean_len": round(mean_len, 2),
        "max_len": max_len,
        "split_events": split_events_total,
        "total_strands": total_strands,
        "unique_seqs": unique_seqs,
        "top1_count": top1_count,
        "redundancy": round(redundancy, 4),
        # "kmer_occupancy": round(occupancy, 4),
        "max_generation": max(generations) if generations else 0,
        "mean_generation": round(sum(generations) / len(generations), 2) if generations else 0.0,
    }


    # Per-k k-mer statistics: extract k-mers from contiguous runs only
    for kval in KMER_RANGE:
        all_kmers = []
        for p in soup:
            for run in top_runs(p):
                all_kmers.extend(extract_kmers(run, kval))
            for run in bottom_runs(p):
                all_kmers.extend(extract_kmers(run, kval))

        total_km = len(all_kmers)
        if total_km == 0:
            result[f"total_kmers_k{kval}"] = 0
            result[f"unique_kmers_k{kval}"] = 0
            result[f"repeat_mass_fraction_k{kval}"] = 0.0
            result[f"n_repeated_types_k{kval}"] = 0
            result[f"top1_kmer_count_k{kval}"] = 0
            result[f"top10_kmer_fraction_k{kval}"] = 0.0
            result[f"redundancy_k{kval}"] = 0.0
            continue

        counter = Counter(all_kmers)
        unique_km = len(counter)

        singleton_occurrences = 0
        n_repeated_types = 0
        top1 = 0
        for _, c in counter.items():
            if c == 1:
                singleton_occurrences += 1
            else:
                n_repeated_types += 1
            if c > top1:
                top1 = c

        repeat_mass_fraction = (total_km - singleton_occurrences) / total_km

        top10_total = 0
        for _, c in counter.most_common(10):
            top10_total += c
        top10_fraction = top10_total / total_km

        redundancy_ratio = 1.0 - (unique_km / total_km)

        result[f"total_kmers_k{kval}"] = total_km
        result[f"unique_kmers_k{kval}"] = unique_km
        result[f"repeat_mass_fraction_k{kval}"] = round(repeat_mass_fraction, 6)
        result[f"n_repeated_types_k{kval}"] = n_repeated_types
        result[f"top1_kmer_count_k{kval}"] = top1
        result[f"top10_kmer_fraction_k{kval}"] = round(top10_fraction, 6)
        result[f"redundancy_k{kval}"] = round(redundancy_ratio, 6)
    return result


# ================================================
# FRAME DUMPING
# ================================================
def soup_frame_text(soup, cycle, phase, max_show=40):
    lines = []
    showing = min(len(soup), max_show)
    lines.append(f"Cycle {cycle} | phase={phase} | pieces={len(soup)} | showing={showing}")
    lines.append("")
    for k, p in enumerate(soup[:max_show], start=1):
        cols = p.cols
        top = " ".join((t if t is not None else "_") for (t, b, _) in cols)
        bot = " ".join((b if b is not None else "_") for (t, b, _) in cols)
        elong = "E" if p.can_elongate else "T"
        sib = f"sib={p.sibling_id}" if p.sibling_id is not None else ""
        lines.append(f"[{k:02d}] id={p.id:<4d} len={len(cols):>2} gen={p.generation} {elong} {sib}")
        lines.append(f"  5' {top} 3'")
        lines.append(f"  3' {bot} 5'")
        lines.append("")
    return "\n".join(lines)


def save_frame_png(text, png_path):
    fig = plt.figure(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor("white")
    plt.axis("off")
    plt.text(0.01, 0.99, text, va="top", ha="left", family="monospace", fontsize=12)
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0.5)
    plt.close(fig)


# ================================================
# SINGLE SIMULATION RUN
# ================================================

def run(seed, allow_pairing, allow_merging, label,
        total_cycles=500, carrying_capacity=360, pool_per_nuc=125,
        k=3, mu=MU, verbose=False,
        save_frames=False, frames_dir="frames", frame_interval=1):
    global NEXT_ID
    NEXT_ID = 1
    random.seed(seed)

    pool = []
    for nuc in NUCLEOTIDES:
        pool.extend([nuc] * pool_per_nuc)
    random.shuffle(pool)

    # Start with empty soup (no seed pieces)
    soup = []

    split_events_total = 0
    tracker = SplitTracker(k=k)
    metrics = []

    if save_frames:
        frame_dir = os.path.join(frames_dir, f"seed_{seed:02d}", label)
        os.makedirs(frame_dir, exist_ok=True)

    def dump_frame(phase_name):
        if not save_frames:
            return
        if cycle % frame_interval != 0:
            return
        text = soup_frame_text(soup, cycle, phase=phase_name)
        png_path = os.path.join(frame_dir, f"cycle_{cycle:04d}_{phase_name}.png")
        save_frame_png(text, png_path)

    for cycle in range(1, total_cycles + 1):
        if not pool:
            break
        nuc = pool.pop()

        if verbose:
            print(f"\n── Cycle {cycle} ── nuc={nuc}, pieces={len(soup)}")

        # Phase 0: Mutation
        mutate_soup(soup, mu=mu)

        # Phase 0: Mismatch-induced splitting (after mutation)
        mismatch_split_soup(soup, cycle)

        # Phase 1: Nucleotide absorption
        absorb_nucleotide(soup, nuc, allow_pairing=allow_pairing,
                          cycle=cycle, verbose=verbose)
        dump_frame("1_absorb")

        # Phase 2: Splitting (complete ds → 2 ss children)
        new_soup = []
        for p in soup:
            if p.is_complete():
                children = split(p, cycle, verbose=verbose)
                if len(children) == 2:
                    split_events_total += 1
                    tracker.log_split(cycle, p, children[0], children[1])
                new_soup.extend(children)
            else:
                new_soup.append(p)
        soup = [p for p in new_soup if not p.is_empty()]
        dump_frame("2_split")

        # Phase 3: Merging (complementary association)
        if allow_merging:
            do_merging_phase(soup, verbose=verbose)
        dump_frame("3_merge")

        # Carrying capacity
        soup = degrade_soup(soup, carrying_capacity)

        # Track child-sequence survival
        tracker.scan_soup(soup, cycle)

        # Metrics
        metrics.append(compute_metrics(soup, cycle, seed, label,
                                       split_events_total, k=k))

    survival_rows = []
    for row in tracker.compute_survival_rows():
        row["seed"] = seed
        row["cond"] = label
        survival_rows.append(row)

    return metrics, survival_rows


# ================================================
# EXPERIMENT RUNNER
# ================================================

def run_experiment(n_seeds=15, total_cycles=500, carrying_capacity=360,
                   pool_per_nuc=125, k=3, mu=MU,
                   metrics_csv="metrics.csv", survival_csv="survival.csv",
                   save_frames=False, frames_dir="frames", frame_interval=1):
    """
    Run 3 conditions × n_seeds and write CSV outputs.

    Conditions:
      1. pair+elong+merge  (pairing=True,  merging=True)
      2. pair+elong        (pairing=True,  merging=False)
      3. elong_only        (pairing=False, merging=False)
    """
    COND_1 = "pair+elong+merge"
    COND_2 = "pair+elong"
    COND_3 = "elong_only"

    configs = [
        (COND_1, True,  True),
        (COND_2, True,  False),
        (COND_3, False, False),
    ]

    all_metrics = []
    all_survival = []
    total_runs = n_seeds * len(configs)
    done = 0

    for label, pairing, merging in configs:
        for s in range(1, n_seeds + 1):
            m, surv = run(
                seed=s, allow_pairing=pairing, allow_merging=merging,
                label=label, total_cycles=total_cycles,
                carrying_capacity=carrying_capacity,
                pool_per_nuc=pool_per_nuc, k=k, mu=mu,
                save_frames=save_frames, frames_dir=frames_dir,
                frame_interval=frame_interval,
            )
            all_metrics.extend(m)
            all_survival.extend(surv)
            done += 1
            if len(m) == 0:
                raise RuntimeError("Simulation produced no metrics rows")
            final = m[-1]
            print(f"  [{done}/{total_runs}] {label:20s} seed={s:2d}  "
                  f"splits={final['split_events']}  "
                  f"pieces={final['pieces']}  "
                  f"mean_len={float(final['mean_len']):.1f}")


    if all_metrics:
        # Fieldnames must be the union of keys across all rows (rows may have condition-dependent columns).
        base_fields = [
            "seed",
            "cycle",
            "cond",
            "pieces",
            "n_ds",
            "mean_len",
            "max_len",
            "split_events",
            "total_strands",
            "unique_seqs",
            "top1_count",
            "redundancy",
            # "kmer_occupancy",
            "max_generation",
            "mean_generation",
        ]
        key_union = set()
        for r in all_metrics:
            for kname in r.keys():
                key_union.add(kname)

        fnames = []
        for kname in base_fields:
            if kname in key_union:
                fnames.append(kname)
                key_union.remove(kname)

        for kname in sorted(key_union):
            fnames.append(kname)

        with open(metrics_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            w.writerows(all_metrics)
        print(f"\nWrote {metrics_csv} ({len(all_metrics)} rows)")


    if all_survival:
        key_union = set()
        for r in all_survival:
            for kname in r.keys():
                key_union.add(kname)
        fnames = sorted(key_union)

        with open(survival_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fnames)
            w.writeheader()
            w.writerows(all_survival)
        print(f"Wrote {survival_csv} ({len(all_survival)} rows)")
    else:
        print("WARNING: No split events recorded.")


# ================================================
# ENTRY POINT
# ================================================
if __name__ == "__main__":
    log_path = "simulation.log"

    with open(log_path, "w", encoding="utf-8") as log_f:
        # with redirect_stdout(log_f), redirect_stderr(log_f):
        with redirect_stdout(log_f):
            print("=" * 60)
            print("Barricelli DNA-norm Simulation")
            print("=" * 60)
            print()

            run_experiment(
                n_seeds=50,
                total_cycles=400,
                carrying_capacity=180,
                pool_per_nuc=100,
                k=5,
                mu=0.0001,
                metrics_csv="metrics.csv",
                survival_csv="survival.csv",
                save_frames=False,
                frames_dir="frames",
                frame_interval=1,
            )

    # This prints to the terminal AFTER redirection ends:
    print(f"Wrote log to {log_path}")