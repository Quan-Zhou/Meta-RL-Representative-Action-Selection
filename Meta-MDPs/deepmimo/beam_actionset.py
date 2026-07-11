import csv
import pickle

import numpy as np
import DeepMIMOv3 as DeepMIMO


def generate_dft_codebook(num_antennas):
    """DFT codebook for a 1D ULA. Shape: (num_antennas, num_beams)."""
    j = 1j
    n = np.arange(num_antennas).reshape(-1, 1)
    k = np.arange(num_antennas).reshape(1, -1)
    return np.exp(-j * 2 * np.pi * n * k / num_antennas) / np.sqrt(num_antennas)


def load_bs_data(user_rows, scenario="asu_campus1", dataset_folder="./scenarios",
                  num_bs_antennas=64, num_subcarriers=64):
    """Loads the fixed, pre-raytraced channel grid for the given row range."""
    parameters = DeepMIMO.default_params()
    parameters["dataset_folder"] = dataset_folder
    parameters["scenario"] = scenario
    parameters["active_BS"] = np.array([1])
    parameters["user_rows"] = user_rows
    parameters["enable_BS2BS"] = 0

    parameters["bs_antenna"]["shape"] = np.array([num_bs_antennas, 1])
    parameters["bs_antenna"]["spacing"] = 0.5
    parameters["ue_antenna"]["shape"] = np.array([1, 1])
    parameters["ue_antenna"]["spacing"] = 0.5

    parameters["OFDM"]["bandwidth"] = 0.05
    parameters["OFDM"]["subcarriers"] = num_subcarriers
    parameters["OFDM"]["selected_subcarriers"] = np.arange(num_subcarriers)

    dataset = DeepMIMO.generate_data(parameters)
    return dataset[0]


def usable_users(bs_data, min_power_ratio=1e-6):
    """Indices of users with a non-negligible OFDM channel.

    LoS != -1 alone is not enough: DeepMIMO clips ray paths whose ToA exceeds the
    useful OFDM symbol duration (visible as its own warning at load time), which
    leaves a real fraction of NLoS users -- 26% for asu_campus1 at 64 subcarriers /
    50MHz -- with LoS != -1 but an exactly-zero channel and hence no meaningful
    "best beam". Filter on actual channel power instead.
    """
    los_ok = bs_data["user"]["LoS"] != -1
    power = np.mean(np.abs(bs_data["user"]["channel"]) ** 2, axis=(1, 2, 3))
    power_ok = power > min_power_ratio * power.max()
    return np.where(los_ok & power_ok)[0]


def geometric_angles_deg(bs_data, user_idx):
    """Azimuth angle in degrees from the BS to each requested user."""
    bs_xy = bs_data["location"][:2]
    user_xy = bs_data["user"]["location"][user_idx, :2]
    delta = user_xy - bs_xy
    return np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))


def best_beams(bs_data, user_idx, codebook):
    """Argmax-gain beam index per user, vectorized over user_idx."""
    channels = np.mean(bs_data["user"]["channel"][user_idx], axis=-1).squeeze(axis=1)
    gains = np.abs(channels.conj() @ codebook)
    return np.argmax(gains, axis=1)


def assign_sections(angles, num_sections, angle_min=None, angle_max=None):
    """Bins angles into num_sections equal-width sections over the observed range."""
    if angle_min is None:
        angle_min = angles.min()
    if angle_max is None:
        angle_max = angles.max()
    edges = np.linspace(angle_min, angle_max, num_sections + 1)
    section_ids = np.clip(np.digitize(angles, edges[1:-1]), 0, num_sections - 1)
    return section_ids, edges


def gaussian_mixture_density(bs_data, user_idx, components):
    """Unnormalized mixture-of-Gaussians density at each user_idx position.

    components: list of {'mean': (x, y), 'cov': scalar sigma^2 or 2x2 array, 'weight': float}
    """
    positions = bs_data["user"]["location"][user_idx, :2]
    density = np.zeros(len(user_idx))
    for comp in components:
        mean = np.asarray(comp["mean"])
        cov = comp["cov"]
        cov = np.eye(2) * cov if np.isscalar(cov) else np.asarray(cov)
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        diff = positions - mean
        exponent = -0.5 * np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        norm_const = 1.0 / (2 * np.pi * np.sqrt(det_cov))
        density += comp["weight"] * norm_const * np.exp(exponent)
    return density


def weighted_permutation(weights, rng):
    """Efraimidis-Spirakis weighted random ordering without replacement (PPSWOR):
    each item gets key u^(1/w), u~Uniform(0,1); sorting by key descending yields a
    sample order consistent with sampling proportional to weight, without replacement."""
    weights = np.clip(np.asarray(weights, dtype=np.float64), 1e-300, None)
    u = rng.random(len(weights))
    keys = u ** (1.0 / weights)
    return np.argsort(-keys)


def draw_training_pairs(idx, bs_data, codebook, rng, weights, train_size):
    """Draws train_size single users and returns (train_idx, positions, true best beams)
    -- the (position, label) pairs an ML beam predictor would train on."""
    order = rng.permutation(len(idx)) if weights is None else weighted_permutation(weights, rng)
    order = order[:train_size]
    train_idx = idx[order]
    train_positions = bs_data["user"]["location"][train_idx, :2]
    train_beams = best_beams(bs_data, train_idx, codebook)
    return train_idx, train_positions, train_beams


def draw_actionset(idx, section_ids, num_sections, bs_data, codebook, rng, weights=None,
                    min_per_section=1, max_draws=100_000):
    """Draws single users (weighted, or uniform if weights is None) one at a time without
    replacement -- the per-user analog of cartpole's "one greedy action per training run,
    unioned into actionset_dict". Stops once every section has >= min_per_section draws,
    or once max_draws is reached (the pool is finite, so this can't loop forever, but
    max_draws bounds the work if a section's weight is very small).

    Returns (actionset_dict, drawn_idx, shortfall) where shortfall lists sections that
    never reached min_per_section within max_draws.
    """
    order = rng.permutation(len(idx)) if weights is None else weighted_permutation(weights, rng)
    order = order[:max_draws]
    ordered_sections = section_ids[order]

    counts = np.zeros(num_sections, dtype=int)
    stop_at = len(order)
    for i, sec in enumerate(ordered_sections):
        counts[sec] += 1
        if np.all(counts >= min_per_section):
            stop_at = i + 1
            break

    used_order = order[:stop_at]
    drawn_idx = idx[used_order]
    drawn_sections = section_ids[used_order]
    drawn_beams = best_beams(bs_data, drawn_idx, codebook)

    actionset_dict = {s: set() for s in range(num_sections)}
    for sec, beam in zip(drawn_sections, drawn_beams):
        actionset_dict[int(sec)].add(int(beam))
    actionset_dict = {s: sorted(b) for s, b in actionset_dict.items()}

    final_counts = np.bincount(drawn_sections, minlength=num_sections)
    shortfall = {s: int(final_counts[s]) for s in range(num_sections) if final_counts[s] < min_per_section}
    print(f"draw_actionset: used {stop_at} draws" + (" (hit max_draws)" if stop_at == len(order) and shortfall else ""))
    return actionset_dict, drawn_idx, shortfall


def draw_actionset_capped(idx, section_ids, num_sections, bs_data, codebook, rng, weights=None,
                           min_per_section=1, max_draws=100_000):
    """New baseline: like draw_actionset, but each section stops CONTRIBUTING to the
    action set once it reaches min_per_section draws -- draws landing in an
    already-capped section are discarded (not added to that section's set), while the
    loop keeps drawing overall until every section reaches quota (or max_draws).

    Fixes the imbalance in draw_actionset where popular sections keep absorbing draws
    (and growing their action set well past M) while the loop waits for the rarest
    section to catch up -- e.g. observed sizes {0: 1, 1: 25, 2: 12, 3: 6} for M=10,
    because sections 1-3 kept accumulating during the 472-draw wait for section 0.
    Here, sections 1-3 would stop growing their action set right at M=10 each, at the
    cost of "wasting" draws on users whose info then gets discarded.

    Returns (actionset_dict, drawn_idx, shortfall). drawn_idx includes ALL probed users
    (including discarded ones) for draw-budget/exclusion accounting.
    """
    order = rng.permutation(len(idx)) if weights is None else weighted_permutation(weights, rng)
    order = order[:max_draws]
    ordered_sections = section_ids[order]

    counts = np.zeros(num_sections, dtype=int)
    contributing_positions = []
    stop_at = len(order)
    for i, sec in enumerate(ordered_sections):
        if counts[sec] < min_per_section:
            counts[sec] += 1
            contributing_positions.append(i)
        if np.all(counts >= min_per_section):
            stop_at = i + 1
            break

    used_order = order[:stop_at]
    drawn_idx = idx[used_order]

    contrib_order = order[contributing_positions]
    contrib_idx = idx[contrib_order]
    contrib_sections = section_ids[contrib_order]
    contrib_beams = best_beams(bs_data, contrib_idx, codebook)

    actionset_dict = {s: set() for s in range(num_sections)}
    for sec, beam in zip(contrib_sections, contrib_beams):
        actionset_dict[int(sec)].add(int(beam))
    actionset_dict = {s: sorted(b) for s, b in actionset_dict.items()}

    shortfall = {s: int(counts[s]) for s in range(num_sections) if counts[s] < min_per_section}
    print(f"draw_actionset_capped: used {stop_at} draws total ({len(contributing_positions)} contributing)"
          + (" (hit max_draws)" if stop_at == len(order) and shortfall else ""))
    return actionset_dict, drawn_idx, shortfall


def evaluate_regret(bs_data, idx, section_ids, codebook, actionset_dict, num_eval, rng,
                     weights=None, exclude_idx=None):
    """Draws num_eval new users (excluding exclude_idx) and computes, per user:
      full_reward       = best gain over the entire codebook (oracle)
      restricted_reward = best gain over just that user's section's action set
      regret            = full_reward - restricted_reward (>= 0; 0 if the action set
                           already contains that user's true optimal beam)
    Users landing in a section with an empty action set get restricted_reward = 0
    (worst case), which shows up as a spike at regret = full_reward.
    """
    if exclude_idx is not None:
        keep = ~np.isin(idx, exclude_idx)
        idx = idx[keep]
        section_ids = section_ids[keep]
        if weights is not None:
            weights = weights[keep]

    order = rng.permutation(len(idx)) if weights is None else weighted_permutation(weights, rng)
    order = order[:num_eval]
    sample_idx = idx[order]
    sample_sections = section_ids[order]

    channels = np.mean(bs_data["user"]["channel"][sample_idx], axis=-1).squeeze(axis=1)
    gains = np.abs(channels.conj() @ codebook)
    full_reward = gains.max(axis=1)

    restricted_reward = np.zeros(len(sample_idx))
    for i, sec in enumerate(sample_sections):
        beams = actionset_dict.get(int(sec), [])
        restricted_reward[i] = gains[i, beams].max() if beams else 0.0

    regret = full_reward - restricted_reward
    return regret, full_reward, restricted_reward, sample_sections, sample_idx


def calibrate_noise_std(full_reward_sample, snr_db):
    """Per-real/imag-component noise std for a target pilot SNR (dB), calibrated to
    this scenario's actual signal scale (mean squared full-codebook gain) rather than
    an arbitrary absolute constant."""
    signal_power = float(np.mean(full_reward_sample ** 2))
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    return np.sqrt(noise_power / 2)


def probe_many(h, candidate_beams, rng, noise_std):
    """Noisy measurement |h^H w + n| for each candidate beam, one probe (one independent
    noise draw) per candidate -- the algorithm never sees h itself, only these scalars.
    candidate_beams: (num_antennas, num_candidates)."""
    signal = h.conj() @ candidate_beams
    noise = rng.normal(0, noise_std, size=signal.shape) + 1j * rng.normal(0, noise_std, size=signal.shape)
    return np.abs(signal + noise)


def search_exhaustive(h, codebook, rng, noise_std):
    """#1: probe every beam in the full codebook once."""
    measurements = probe_many(h, codebook, rng, noise_std)
    chosen = int(np.argmax(measurements))
    return chosen, codebook.shape[1]


def search_actionset(h, beams, codebook, rng, noise_std):
    """#4 (ours): probe only the beams in this user's section's action set."""
    if not beams:
        return None, 0
    measurements = probe_many(h, codebook[:, beams], rng, noise_std)
    chosen = beams[int(np.argmax(measurements))]
    return chosen, len(beams)


def search_hierarchical(h, codebook, rng, noise_std, coarse_stride):
    """#2: probe coarse_stride-spaced beams first (coarse sweep), then probe the
    coarse_stride fine beams inside whichever coarse beam won."""
    num_beams = codebook.shape[1]
    coarse_idx = list(range(0, num_beams, coarse_stride))
    coarse_measurements = probe_many(h, codebook[:, coarse_idx], rng, noise_std)
    best_coarse = coarse_idx[int(np.argmax(coarse_measurements))]
    fine_idx = list(range(best_coarse, min(best_coarse + coarse_stride, num_beams)))
    fine_measurements = probe_many(h, codebook[:, fine_idx], rng, noise_std)
    chosen = fine_idx[int(np.argmax(fine_measurements))]
    return chosen, len(coarse_idx) + len(fine_idx)


def make_cs_sensing_matrix(num_antennas, num_probes, rng):
    """M random phase-only probe vectors (unit norm) -- realistic for hybrid arrays
    with phase-shifter-only analog combining, no amplitude control needed."""
    phases = rng.uniform(0, 2 * np.pi, size=(num_antennas, num_probes))
    return np.exp(1j * phases) / np.sqrt(num_antennas)


def probe_complex_many(h, candidate_beams, rng, noise_std):
    """Like probe_many, but returns the complex (phase-coherent) measurement instead of
    just its magnitude. Needed for compressed-sensing recovery (OMP needs linear
    measurements); realistic for a receiver with at least one coherent RF chain."""
    signal = h.conj() @ candidate_beams
    noise = rng.normal(0, noise_std, size=signal.shape) + 1j * rng.normal(0, noise_std, size=signal.shape)
    return signal + noise


def omp_complex(sensing_matrix, y, sparsity):
    """Orthogonal Matching Pursuit for complex-valued sparse recovery: y ~ sensing_matrix @ x,
    x sparse. sensing_matrix: (M, N), y: (M,). Returns x_hat: (N,)."""
    num_measurements, num_atoms = sensing_matrix.shape
    residual = y.copy()
    support = []
    coeffs = np.zeros(0, dtype=complex)
    for _ in range(min(sparsity, num_measurements)):
        correlations = np.abs(sensing_matrix.conj().T @ residual)
        correlations[support] = -np.inf
        support.append(int(np.argmax(correlations)))
        A_support = sensing_matrix[:, support]
        coeffs, *_ = np.linalg.lstsq(A_support, y, rcond=None)
        residual = y - A_support @ coeffs
    x_hat = np.zeros(num_atoms, dtype=complex)
    x_hat[support] = coeffs
    return x_hat


def search_compressed_sensing(h, codebook, phi, rng, noise_std, sparsity):
    """#3: probe with M random combined (phase-only) beams, recover a sparse estimate of
    h in the codebook's angular basis via OMP, then pick the argmax beam of the
    RECONSTRUCTED channel (free -- no extra physical probes once you have the estimate).

    Model: treat q := h.conj() as approximately codebook @ x for sparse x (reasonable
    since a single codebook column already captures most of a real channel's gain --
    that's what makes DFT-codebook beam selection work at all). Measurements
    y = q @ phi ~= x @ (codebook.T @ phi), so the sensing matrix for OMP is
    A = (codebook.T @ phi).T = phi.T @ codebook, recovering x from y.
    """
    y = probe_complex_many(h, phi, rng, noise_std)
    sensing_matrix = phi.T @ codebook
    x_hat = omp_complex(sensing_matrix, y, sparsity)
    q_hat = codebook @ x_hat  # estimate of h.conj()
    estimated_gains = np.abs(q_hat @ codebook)
    chosen = int(np.argmax(estimated_gains))
    return chosen, phi.shape[1]


def build_ml_beam_prior(train_positions, train_beams, num_beams, grid_bins):
    """#5 training step (done once, offline): learn P(beam | grid cell) as a simple
    frequency table over a grid_bins x grid_bins spatial grid."""
    x_min, y_min = train_positions.min(axis=0)
    x_max, y_max = train_positions.max(axis=0)
    x_edges = np.linspace(x_min, x_max, grid_bins + 1)
    y_edges = np.linspace(y_min, y_max, grid_bins + 1)

    cell_x = np.clip(np.digitize(train_positions[:, 0], x_edges[1:-1]), 0, grid_bins - 1)
    cell_y = np.clip(np.digitize(train_positions[:, 1], y_edges[1:-1]), 0, grid_bins - 1)

    cell_counts = {}
    for cx, cy, beam in zip(cell_x, cell_y, train_beams):
        key = (int(cx), int(cy))
        if key not in cell_counts:
            cell_counts[key] = np.zeros(num_beams, dtype=int)
        cell_counts[key][int(beam)] += 1

    global_counts = np.bincount(train_beams, minlength=num_beams)  # fallback for cells with no training data
    return {"x_edges": x_edges, "y_edges": y_edges, "grid_bins": grid_bins,
            "cell_counts": cell_counts, "global_counts": global_counts}


def ml_prior_shortlist(query_pos, prior, top_k):
    """#5 inference step: O(1) grid-cell lookup (not a search over training data), then
    rank that cell's learned beam frequencies. Always returns exactly top_k beams for a
    probe-matched comparison: if a cell has fewer than top_k beams with nonzero
    frequency, it's padded with remaining beams in index order -- those padding slots
    are NOT evidence-based, just filler to reach the fixed probe budget."""
    grid_bins = prior["grid_bins"]
    cx = int(np.clip(np.digitize(query_pos[0], prior["x_edges"][1:-1]), 0, grid_bins - 1))
    cy = int(np.clip(np.digitize(query_pos[1], prior["y_edges"][1:-1]), 0, grid_bins - 1))
    counts = prior["cell_counts"].get((cx, cy), prior["global_counts"])
    ranked = np.argsort(-counts)  # nonzero-frequency beams first, then zero-frequency padding by index
    return ranked[:top_k].tolist()


def search_ml_topk(h, shortlist, codebook, rng, noise_std):
    """#5 verification step: probe only the ML-predicted shortlist, same mechanics as
    search_actionset but with a learned candidate set instead of an angular-section one."""
    if not shortlist:
        return None, 0
    measurements = probe_many(h, codebook[:, shortlist], rng, noise_std)
    chosen = shortlist[int(np.argmax(measurements))]
    return chosen, len(shortlist)


def evaluate_search_methods(bs_data, sample_idx, sample_sections, codebook, actionset_dict,
                             rng, noise_std, coarse_stride, phi=None, cs_sparsity=None,
                             ml_prior=None, top_k=None):
    """Runs #1 exhaustive, #4 context-aided, #2 hierarchical, and (if their args are
    given) #3 compressed sensing and #5 ML top-K search -- each through the noisy
    probe() interface -- on the same held-out users, and scores each choice against the
    TRUE (noiseless) channel: selection uses noisy pilots, but the achieved reward is
    whatever the true channel actually gives you with the beam you landed on."""
    methods = ["exhaustive", "actionset", "hierarchical"]
    if phi is not None:
        methods.append("compressed_sensing")
    if ml_prior is not None:
        methods.append("ml_topk")
    results = {m: {"regret": [], "num_probes": []} for m in methods}

    for i, sec in zip(sample_idx, sample_sections):
        h = np.mean(bs_data["user"]["channel"][i], axis=-1).squeeze()
        true_gains = np.abs(h.conj() @ codebook)
        true_best = true_gains.max()

        chosen, n = search_exhaustive(h, codebook, rng, noise_std)
        results["exhaustive"]["regret"].append(true_best - true_gains[chosen])
        results["exhaustive"]["num_probes"].append(n)

        beams = actionset_dict.get(int(sec), [])
        chosen, n = search_actionset(h, beams, codebook, rng, noise_std)
        achieved = true_gains[chosen] if chosen is not None else 0.0
        results["actionset"]["regret"].append(true_best - achieved)
        results["actionset"]["num_probes"].append(n)

        chosen, n = search_hierarchical(h, codebook, rng, noise_std, coarse_stride)
        results["hierarchical"]["regret"].append(true_best - true_gains[chosen])
        results["hierarchical"]["num_probes"].append(n)

        if phi is not None:
            chosen, n = search_compressed_sensing(h, codebook, phi, rng, noise_std, cs_sparsity)
            results["compressed_sensing"]["regret"].append(true_best - true_gains[chosen])
            results["compressed_sensing"]["num_probes"].append(n)

        if ml_prior is not None:
            query_pos = bs_data["user"]["location"][i, :2]
            shortlist = ml_prior_shortlist(query_pos, ml_prior, top_k)
            chosen, n = search_ml_topk(h, shortlist, codebook, rng, noise_std)
            achieved = true_gains[chosen] if chosen is not None else 0.0
            results["ml_topk"]["regret"].append(true_best - achieved)
            results["ml_topk"]["num_probes"].append(n)

    summary = {}
    for m in methods:
        regret = np.array(results[m]["regret"])
        summary[m] = {
            "mean_regret": float(regret.mean()),
            "median_regret": float(np.median(regret)),
            "max_regret": float(regret.max()),
            "exact_match_rate": float(np.mean(regret == 0.0)),
            "mean_num_probes": float(np.mean(results[m]["num_probes"])),
        }
    return summary


def save_search_comparison_table(search_summary, path="search_comparison.csv"):
    """Saves the #1/#4/#2 noisy-search comparison (one row per method) to CSV."""
    fieldnames = ["method", "mean_num_probes", "exact_match_rate",
                  "mean_regret", "median_regret", "max_regret"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for method, stats in search_summary.items():
            writer.writerow({"method": method, **{k: stats[k] for k in fieldnames if k != "method"}})
    print(f"Saved {path}")


def run_actionset_and_regret(min_per_section, idx, section_ids, bs_data, codebook,
                              num_sections, max_draws, num_eval, weights, seed, verbose=True,
                              capped=False):
    """One (min_per_section) configuration: build the action sets, then evaluate regret
    on a held-out draw. Returns a summary dict (used for the MIN_PER_SECTION sweep).
    capped=True uses draw_actionset_capped (the per-section-cap baseline) instead of
    draw_actionset."""
    rng = np.random.default_rng(seed)
    draw_fn = draw_actionset_capped if capped else draw_actionset
    actionset_dict, drawn_idx, shortfall = draw_fn(
        idx, section_ids, num_sections, bs_data, codebook, rng, weights=weights,
        min_per_section=min_per_section, max_draws=max_draws,
    )
    regret, full_reward, restricted_reward, sample_sections, sample_idx = evaluate_regret(
        bs_data, idx, section_ids, codebook, actionset_dict, num_eval, rng,
        weights=weights, exclude_idx=drawn_idx,
    )
    num_draws_used = len(drawn_idx)
    avg_actionset_size = float(np.mean([len(b) for b in actionset_dict.values()]))
    summary = {
        "min_per_section": min_per_section,
        "num_draws_used": num_draws_used,
        "avg_actionset_size": avg_actionset_size,
        "shortfall": shortfall,
        "mean_regret": float(regret.mean()),
        "median_regret": float(np.median(regret)),
        "max_regret": float(regret.max()),
        "exact_match_rate": float(np.mean(regret == 0.0)),
    }
    if verbose:
        print(f"  M={min_per_section:>4}: draws={num_draws_used:>6}, avg action-set size={avg_actionset_size:5.1f}/{codebook.shape[1]}, "
              f"exact-match={summary['exact_match_rate']:.1%}, mean regret={summary['mean_regret']:.3e}"
              + (f", SHORTFALL={shortfall}" if shortfall else ""))
    return summary, actionset_dict, regret


def plot_regret_vs_min_per_section(summaries, path="regret_vs_min_per_section.png"):
    """Two small multiples sharing an x-axis (never dual-axis): exact-match rate and
    mean regret, both vs. MIN_PER_SECTION, to show the draws-vs-regret trade-off."""
    import matplotlib.pyplot as plt

    m = [s["min_per_section"] for s in summaries]
    exact_match = [s["exact_match_rate"] * 100 for s in summaries]
    mean_regret = [s["mean_regret"] for s in summaries]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(m, exact_match, marker="o", color="#3B6FE0")
    ax1.set_xscale("log")
    ax1.set_xlabel("MIN_PER_SECTION")
    ax1.set_ylabel("Exact-match rate (%)")
    ax1.set_title("Action-set coverage")
    ax1.spines[["top", "right"]].set_visible(False)

    ax2.plot(m, mean_regret, marker="o", color="#3B6FE0")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("MIN_PER_SECTION")
    ax2.set_ylabel("Mean regret (linear |h^H w| units)")
    ax2.set_title("Mean regret")
    ax2.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Reward comparison: draw budget (MIN_PER_SECTION) vs. regret")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


def save_summary_table(summaries, path="regret_summary.csv"):
    """Saves the MIN_PER_SECTION sweep summary (one row per config) to CSV."""
    fieldnames = ["min_per_section", "num_draws_used", "avg_actionset_size",
                  "exact_match_rate", "mean_regret", "median_regret", "max_regret", "shortfall"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in summaries:
            writer.writerow({k: s[k] for k in fieldnames})
    print(f"Saved {path}")


def plot_regret_distribution(regret, min_per_section, num_draws_used, path="regret_distribution.png"):
    """Histogram of regret (full-codebook gain - action-set gain), log-scale y-axis so
    the tail stays visible behind the large exact-zero-regret spike, with that spike
    called out explicitly."""
    import matplotlib.pyplot as plt

    zero_frac = float(np.mean(regret == 0.0))
    nonzero = regret[regret > 0.0]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(regret, bins=40, color="#3B6FE0", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    ax.set_xlabel("Regret (full-codebook gain − action-set gain), linear |h^H w| units")
    ax.set_ylabel("Number of evaluation users (log scale)")
    ax.set_title(f"Per-user regret of the reduced per-section action set\n"
                 f"(MIN_PER_SECTION={min_per_section}, draws used={num_draws_used})")
    ax.annotate(f"{zero_frac:.1%} of users: exact match\n(zero regret)",
                xy=(0, len(regret) * zero_frac), xytext=(0.15, 0.85),
                textcoords="axes fraction", fontsize=9, color="#333333",
                arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))
    if len(nonzero) > 0:
        ax.text(0.98, 0.98, f"nonzero regret: median={np.median(nonzero):.2e}, "
                             f"max={nonzero.max():.2e}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color="#333333")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    NUM_SECTIONS = 4
    MIN_PER_SECTION = 1    # keep drawing until every section has >= this many draws
    MAX_NUM_DRAWS = 100_000  # safety cap on the draw-until-covered loop (pool is finite, so this can't hang, but bounds the work)
    NUM_EVAL = 10000       # held-out single-user draws used to measure regret
    USER_ROWS = np.arange(1, 101)  # first 100 of 321 rows; widen if action sets/eval come up short
    NUM_BS_ANTENNAS = 128  # also sets full codebook size (generate_dft_codebook ties num beams to num antennas)
    SEED = 0
    NUM_HOTSPOTS = 3
    HOTSPOT_SIGMA_FRACTION = 0.1  # hotspot std, as a fraction of the position bounding-box diagonal

    print("Loading DeepMIMO grid...")
    bs_data = load_bs_data(USER_ROWS, num_bs_antennas=NUM_BS_ANTENNAS)
    codebook = generate_dft_codebook(NUM_BS_ANTENNAS)

    idx = usable_users(bs_data)
    print(f"Loaded {len(bs_data['user']['LoS'])} raw users, {len(idx)} usable "
          f"(LoS != -1 and non-negligible channel power)")

    angles = geometric_angles_deg(bs_data, idx)
    section_ids, edges = assign_sections(angles, NUM_SECTIONS)
    print(f"Angle range: [{edges[0]:.1f}, {edges[-1]:.1f}] deg, edges: {np.round(edges, 1)}")

    rng = np.random.default_rng(SEED)

    # Hotspot centers are real grid points (guarantees they sit somewhere the scenario
    # actually has), sigma scaled to the scenario's spatial extent.
    positions = bs_data["user"]["location"][idx, :2]
    diag = np.linalg.norm(positions.max(axis=0) - positions.min(axis=0))
    sigma = diag * HOTSPOT_SIGMA_FRACTION
    hotspot_pos_idx = rng.choice(len(idx), size=NUM_HOTSPOTS, replace=False)
    components = [
        {"mean": positions[i], "cov": sigma ** 2, "weight": 1.0 / NUM_HOTSPOTS}
        for i in hotspot_pos_idx
    ]
    print(f"Hotspots (sigma={sigma:.1f}m): {[tuple(np.round(c['mean'], 1)) for c in components]}")

    weights = gaussian_mixture_density(bs_data, idx, components)
    eff_sample_size = weights.sum() ** 2 / np.sum(weights ** 2)
    print(f"Effective sample size under hotspot weighting: {eff_sample_size:.0f} / {len(idx)} usable users")

    print("Regret units: linear beam-gain amplitude |h^H w| (unit-norm codebook, "
          "DeepMIMO channel h already includes path loss/phase/array response) -- "
          "not power, not dB.")

    # Reward-comparison wrap-up: sweep MIN_PER_SECTION to see the draws-vs-regret
    # trade-off, instead of just the single MIN_PER_SECTION=1 config from before.
    MIN_PER_SECTION_SWEEP = [1, 3, 10, 30, 100, 300]
    DISTRIBUTION_PLOT_M = 10  # which sweep config's regret distribution / action set to use downstream
    print(f"Sweeping MIN_PER_SECTION over {MIN_PER_SECTION_SWEEP} "
          f"(NUM_SECTIONS={NUM_SECTIONS}, NUM_EVAL={NUM_EVAL}, MAX_NUM_DRAWS={MAX_NUM_DRAWS}):")
    summaries = []
    actionset_dict_by_m = {}
    regret_by_m = {}
    for m in MIN_PER_SECTION_SWEEP:
        summary, actionset_dict, regret = run_actionset_and_regret(
            m, idx, section_ids, bs_data, codebook, NUM_SECTIONS, MAX_NUM_DRAWS,
            NUM_EVAL, weights, seed=SEED,
        )
        summaries.append(summary)
        actionset_dict_by_m[m] = actionset_dict
        regret_by_m[m] = regret

    with open("beam_actionset.pkl", "wb") as f:
        pickle.dump({"actionset_dict": actionset_dict_by_m[MIN_PER_SECTION_SWEEP[-1]], "edges": edges,
                     "sweep_summaries": summaries}, f)
    print("Saved beam_actionset.pkl (action set from the largest MIN_PER_SECTION run)")

    save_summary_table(summaries)
    plot_regret_vs_min_per_section(summaries)
    distribution_summary = next(s for s in summaries if s["min_per_section"] == DISTRIBUTION_PLOT_M)
    plot_regret_distribution(regret_by_m[DISTRIBUTION_PLOT_M], min_per_section=DISTRIBUTION_PLOT_M,
                              num_draws_used=distribution_summary["num_draws_used"])

    # Runtime comparison: #1 exhaustive, #4 context-aided (ours), #2 hierarchical,
    # #3 compressed sensing, #5 ML top-K. Every method decides using only noisy probes
    # (h unknown to the algorithm); evaluation is allowed to use the true h to score
    # what was actually achieved.
    SNR_DB = 10 
    COARSE_STRIDE = 8
    CS_NUM_PROBES = 16      # M random combined probes for compressed sensing
    CS_SPARSITY = 3         # assumed number of dominant paths for OMP
    ML_TRAIN_SIZE = distribution_summary["num_draws_used"]  # match our method's data budget exactly
    ML_GRID_BINS = 6        # spatial grid resolution for the learned beam-frequency prior (6x6 = 36 cells)
    ML_TOP_K = 15           # shortlist size probed after prediction (matched to our ~14.5-probe action set)
    SEARCH_NUM_EVAL = 10000  # this loop isn't vectorized (independent noise per probe), so it's slower than NUM_EVAL

    runtime_actionset = actionset_dict_by_m[DISTRIBUTION_PLOT_M]

    _, train_positions, train_beams = draw_training_pairs(
        idx, bs_data, codebook, rng, weights, ML_TRAIN_SIZE
    )
    ml_prior = build_ml_beam_prior(train_positions, train_beams, NUM_BS_ANTENNAS, ML_GRID_BINS)

    _, calib_full_reward, _, search_sample_sections, search_sample_idx = evaluate_regret(
        bs_data, idx, section_ids, codebook, runtime_actionset, SEARCH_NUM_EVAL, rng,
        weights=weights,
    )
    noise_std = calibrate_noise_std(calib_full_reward, SNR_DB)
    print(f"Runtime comparison: SNR={SNR_DB}dB -> noise_std={noise_std:.2e} per I/Q component "
          f"(calibrated to mean squared full-codebook gain)")

    phi = make_cs_sensing_matrix(NUM_BS_ANTENNAS, CS_NUM_PROBES, rng)

    search_summary = evaluate_search_methods(
        bs_data, search_sample_idx, search_sample_sections, codebook, runtime_actionset,
        rng, noise_std, COARSE_STRIDE, phi=phi, cs_sparsity=CS_SPARSITY,
        ml_prior=ml_prior, top_k=ML_TOP_K,
    )
    print(f"Noisy search comparison (MIN_PER_SECTION={DISTRIBUTION_PLOT_M} action set, "
          f"{SEARCH_NUM_EVAL} eval users, ML trained on {ML_TRAIN_SIZE} pairs):")
    for method, stats in search_summary.items():
        print(f"  {method:>18}: probes={stats['mean_num_probes']:6.1f}, "
              f"exact-match={stats['exact_match_rate']:.1%}, mean regret={stats['mean_regret']:.3e}")
    save_search_comparison_table(search_summary)
