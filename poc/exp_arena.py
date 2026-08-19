"""
Esperimento 3 - Red team sulla classifica arena (dati REALI LMArena).

Dataset: lmsys/lmsys-arena-human-preference-55k (57.477 battaglie reali,
64 modelli, voti umani del periodo 2023-24, rilasciato pubblicamente).

3a. BALLOT STUFFING
    Un attaccante inietta n voti falsi a favore di un modello bersaglio di
    meta' classifica (vince ogni battaglia iniettata, avversari campionati
    dalla distribuzione reale). Misuriamo le posizioni guadagnate al
    crescere di n, ricalcolando il Bradley-Terry sull'intero dataset.
    DIFESA modellata: proof-of-personhood + cap per identita' (5 voti) +
    down-weighting (0.3) delle identita' nuove monolaterali; costo di una
    identita' verificata: 3 euro (assunzione dichiarata).

3b. VARIANT SHOPPING (selective reporting)
    Un lab esegue N run privati dello stesso modello su un benchmark statico
    da 240 item e pubblica solo il migliore. Quante posizioni si comprano?
    (Con la pre-registrazione il valore e' 0 per costruzione.)

Output: out/results_arena.json + out/arrays_arena.npz
"""
import json
import numpy as np
import pandas as pd
from engine import fit_bt, bt_to_elo, rank_of

SEED = 20260818
N_SWEEP = [125, 250, 500, 1000, 2000, 4000]
MC = 6
DEF_WEIGHT = 0.3      # peso dei voti da identita' nuove monolaterali
CAP_PER_ID = 5        # voti massimi contati per identita' verificata
COST_PER_ID = 3.0     # euro per identita' verificata (proof-of-personhood)
BOOT = 30


def load_votes():
    df = pd.read_parquet("../out/arena_votes.parquet")
    models = sorted(set(df.model_a) | set(df.model_b))
    m2i = {m: i for i, m in enumerate(models)}
    ia = df.model_a.map(m2i).to_numpy()
    ib = df.model_b.map(m2i).to_numpy()
    sa = np.where(df.winner_model_a.to_numpy() == 1, 1.0,
                  np.where(df.winner_model_b.to_numpy() == 1, 0.0, 0.5))
    return models, ia, ib, sa


def positions_curve(ia, ib, sa, n_models, target, opp_pool, s_base, rng, defended):
    """Posizioni ed Elo guadagnati dal target per ogni n in N_SWEEP."""
    base_elo = bt_to_elo(s_base)
    base_rank = rank_of(base_elo)[target]
    gains = np.zeros((len(N_SWEEP), MC))
    elo_gains = np.zeros((len(N_SWEEP), MC))
    for j in range(MC):
        opps = rng.choice(opp_pool, max(N_SWEEP))
        for k, n in enumerate(N_SWEEP):
            fa = np.full(n, target)
            fb = opps[:n]
            w_org = np.ones(len(ia))
            w_fake = np.full(n, DEF_WEIGHT if defended else 1.0)
            s = fit_bt(np.concatenate([ia, fa]), np.concatenate([ib, fb]),
                       np.concatenate([sa, np.ones(n)]), n_models,
                       weights=np.concatenate([w_org, w_fake]), x0=s_base)
            e = bt_to_elo(s)
            gains[k, j] = base_rank - rank_of(e)[target]
            elo_gains[k, j] = e[target] - base_elo[target]
    return base_rank, gains, elo_gains


def interp_votes_for_gain(gains_mean, goal=2.0):
    """Interpola i voti necessari per guadagnare `goal` posizioni."""
    x = np.array(N_SWEEP, dtype=float)
    y = gains_mean
    if y.max() < goal:
        return None
    i = int(np.argmax(y >= goal))
    if i == 0:
        return float(x[0] * goal / max(y[0], 1e-9))
    x0, x1, y0, y1 = x[i-1], x[i], y[i-1], y[i]
    return float(x0 + (goal - y0) * (x1 - x0) / max(y1 - y0, 1e-9))


def variant_shopping(rng):
    """Posizioni comprate pubblicando il migliore di N run (dati da exp1)."""
    arr = np.load("../out/arrays_static.npz")
    acc = arr["acc240"]                     # (R, 20) re-run reali del simulatore
    R, M = acc.shape
    med = np.median(acc, axis=0)
    mid_models = np.argsort(med)[5:15]      # modelli di meta' classifica
    out = {}
    for N in [1, 5, 10, 20]:
        gains = []
        for _ in range(3000):
            m = rng.choice(mid_models)
            best = acc[rng.choice(R, N, replace=False), m].max()
            board = med.copy()
            board[m] = best
            gains.append(rank_of(med)[m] - rank_of(board)[m])
        out[N] = (float(np.mean(gains)), float(np.std(gains)))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    models, ia, ib, sa = load_votes()
    n_models = len(models)
    n_votes = len(ia)

    # --- baseline sui voti reali
    s_base = fit_bt(ia, ib, sa, n_models)
    elo = bt_to_elo(s_base)
    ranks = rank_of(elo)
    counts = np.bincount(np.concatenate([ia, ib]), minlength=n_models)

    # bootstrap CI del rating del target
    order = np.argsort(ranks)
    # target: rank piu' vicino a 30 con almeno 800 battaglie
    cands = [i for i in range(n_models) if counts[i] >= 800]
    target = min(cands, key=lambda i: abs(int(ranks[i]) - 30))
    boots = []
    for _ in range(BOOT):
        idx = rng.integers(0, n_votes, n_votes)
        s_b = fit_bt(ia[idx], ib[idx], sa[idx], n_models, x0=s_base)
        boots.append((bt_to_elo(s_b)[target], int(rank_of(bt_to_elo(s_b))[target])))
    boot_elo = np.array([b[0] for b in boots])
    boot_rank = np.array([b[1] for b in boots])

    # --- attacco senza e con difese
    opp_pool = np.concatenate([ib[ia == target], ia[ib == target]])
    base_rank, g_free, e_free = positions_curve(ia, ib, sa, n_models, target, opp_pool,
                                                s_base, np.random.default_rng(SEED + 1), defended=False)
    _, g_def, e_def = positions_curve(ia, ib, sa, n_models, target, opp_pool,
                                      s_base, np.random.default_rng(SEED + 1), defended=True)

    gm_free, gm_def = g_free.mean(1), g_def.mean(1)
    em_free, em_def = e_free.mean(1), e_def.mean(1)
    v2_free = interp_votes_for_gain(gm_free, 2.0)
    v2_def = interp_votes_for_gain(gm_def, 2.0)
    cost_def = None if v2_def is None else float(np.ceil(v2_def / CAP_PER_ID) * COST_PER_ID)

    # manipolazione OLTRE l'incertezza dichiarata: spostare l'Elo di piu'
    # dell'ampiezza dell'intervallo di confidenza al 95% (ranking a fasce CI)
    ci95_half = 1.96 * float(boot_elo.std())

    def votes_for_elo(em, goal):
        x = np.array(N_SWEEP, dtype=float)
        if em.max() < goal:
            return None
        i = int(np.argmax(em >= goal))
        if i == 0:
            return float(x[0] * goal / max(em[0], 1e-9))
        return float(x[i-1] + (goal - em[i-1]) * (x[i] - x[i-1]) / max(em[i] - em[i-1], 1e-9))

    v_ci_free = votes_for_elo(em_free, 2 * ci95_half)
    v_ci_def = votes_for_elo(em_def, 2 * ci95_half)
    cost_ci_def = None if v_ci_def is None else float(np.ceil(v_ci_def / CAP_PER_ID) * COST_PER_ID)

    vs = variant_shopping(np.random.default_rng(SEED + 2))

    out = {
        "data_source": "lmsys/lmsys-arena-human-preference-55k (voti reali LMArena 2023-24)",
        "n_votes": int(n_votes), "n_models": int(n_models),
        "target": {"name": models[target], "rank": int(base_rank),
                   "elo": float(elo[target]), "battles": int(counts[target]),
                   "boot_elo_sd": float(boot_elo.std()),
                   "boot_rank_range": [int(boot_rank.min()), int(boot_rank.max())]},
        "top10": [{"rank": int(ranks[i]), "model": models[i], "elo": round(float(elo[i]), 1)}
                  for i in order[:10]],
        "sweep": N_SWEEP,
        "gain_free_mean": gm_free.tolist(), "gain_free_sd": g_free.std(1).tolist(),
        "gain_def_mean": gm_def.tolist(), "gain_def_sd": g_def.std(1).tolist(),
        "elo_gain_free_mean": em_free.tolist(), "elo_gain_def_mean": em_def.tolist(),
        "votes_for_2pos_free": v2_free,
        "votes_for_2pos_free_pct": None if v2_free is None else round(100 * v2_free / n_votes, 2),
        "votes_for_2pos_defended": v2_def,
        "cost_2pos_defended_eur": cost_def,
        "ci95_halfwidth_elo": ci95_half,
        "votes_beyond_ci_free": v_ci_free,
        "votes_beyond_ci_defended": v_ci_def,
        "cost_beyond_ci_defended_eur": cost_ci_def,
        "defense_assumptions": {"weight": DEF_WEIGHT, "cap_per_identity": CAP_PER_ID,
                                "cost_per_identity_eur": COST_PER_ID},
        "variant_shopping": {str(N): {"mean_pos": v[0], "sd": v[1]} for N, v in vs.items()},
    }
    with open("../out/results_arena.json", "w") as fh:
        json.dump(out, fh, indent=1)
    np.savez_compressed("../out/arrays_arena.npz",
                        sweep=np.array(N_SWEEP),
                        g_free=g_free, g_def=g_def,
                        e_free=e_free, e_def=e_def)

    print("dataset:", out["data_source"], "| voti:", n_votes, "| modelli:", n_models)
    print("target:", models[target], "| rank", int(base_rank), "| elo %.0f" % elo[target],
          "| battaglie", int(counts[target]), "| boot rank", out["target"]["boot_rank_range"])
    print("posizioni guadagnate (senza difese):", dict(zip(N_SWEEP, np.round(gm_free, 2))))
    print("posizioni guadagnate (con difese):  ", dict(zip(N_SWEEP, np.round(gm_def, 2))))
    print("voti per +2 posizioni: liberi=%s (%.2f%% del totale)  difesi=%s  costo=%s EUR" % (
        None if v2_free is None else int(v2_free), 0 if v2_free is None else 100 * v2_free / n_votes,
        None if v2_def is None else int(v2_def), cost_def))
    print("CI95 half-width Elo target: %.1f | voti per superare la fascia CI: liberi=%s difesi=%s costo=%s EUR" % (
        ci95_half,
        None if v_ci_free is None else int(v_ci_free),
        None if v_ci_def is None else int(v_ci_def), cost_ci_def))
    print("guadagno Elo (liberi):", dict(zip(N_SWEEP, np.round(em_free, 1))))
    print("guadagno Elo (difesi):", dict(zip(N_SWEEP, np.round(em_def, 1))))
    print("variant shopping (posizioni comprate):", {N: round(v[0], 2) for N, v in vs.items()})
