"""
Esperimenti 1 e 2 del PoC.

ESPERIMENTO 1 - Efficienza e saturazione
  Confronto a parita' di budget di domande tra:
    A) accuracy classica su set fisso (status quo dei benchmark statici)
    B) scoring IRT-EAP sullo stesso set fisso (ablazione: solo statistica migliore)
    C) IRT adattivo / CAT (statistica migliore + selezione adattiva)
  Piu' demo di saturazione: set "invecchiato" (difficolta' bassa, stile benchmark
  del 2020 usato nel 2026) vs CAT su banca a piena gamma.

ESPERIMENTO 2 - Contaminazione con difese honeypot + rotazione
  6 modelli su 20 hanno memorizzato una frazione f del set pubblico trapelato.
  Nel set trapelato, 20 item su 200 sono HONEYPOT: la chiave di risposta
  pubblicata e' deliberatamente sbagliata. Chi ha memorizzato riproduce la
  risposta piantata (impronta digitale della memorizzazione).
  Rilevamento: z honeypot + z gap pubblico-vs-fresco (probe fresco da 200 item
  = il set del ciclo successivo, come nella rotazione reale del servizio).
  Correzione: i flaggati vengono ri-classificati sugli item freschi.

Output: out/results_static.json + out/arrays_static.npz
"""
import json
import numpy as np
from engine import (make_bank, make_legacy_bank, EAPGrid, simulate_static,
                    eap_from_static, run_cat, kendall_tau, adjacent_inversions,
                    rank_of, ItemBank, p_correct)

MASTER_SEED = 20260818
N_MODELS = 20
THETA = np.linspace(-2.2, 3.0, N_MODELS)          # campo di 20 modelli, gap ~0.27
TOP5 = np.argsort(THETA)[-5:]                     # i 5 di frontiera
BUDGETS = [30, 60, 120, 240]
R1 = 200
R2 = 300


# ----------------------------------------------------------------- exp 1

def experiment_1(rng):
    bank = make_bank(600, rng)                                  # banca moderna
    legacy = make_legacy_bank(300, rng, mu=-1.2, sd=0.8, b_cap=0.4)  # invecchiata
    eap = EAPGrid(bank)

    tau = {arm: {K: [] for K in BUDGETS} for arm in "ABC"}
    inv = {arm: {K: [] for K in BUDGETS} for arm in "ABC"}
    ciw, cover = {K: [] for K in BUDGETS}, {K: [] for K in BUDGETS}
    acc240 = []
    sat = {"tau5_legacy": [], "tau5_static200": [], "tau5_cat60": [],
           "top1_legacy": [], "top1_static200": [], "top1_cat60": [],
           "accs_top5_legacy": []}

    for r in range(R1):
        for K in BUDGETS:
            items = rng.choice(bank.n, K, replace=False)
            U = simulate_static(bank, THETA, items, rng)
            acc = U.mean(1)
            tau["A"][K].append(kendall_tau(THETA, acc))
            inv["A"][K].append(adjacent_inversions(THETA, acc))
            m, sd, lo, hi = eap_from_static(eap, items, U)
            tau["B"][K].append(kendall_tau(THETA, m))
            inv["B"][K].append(adjacent_inversions(THETA, m))
            if K == 240:
                acc240.append(acc)

        res = run_cat(bank, eap, THETA, BUDGETS, rng)
        for K in BUDGETS:
            m, sd, lo, hi = res[K]
            tau["C"][K].append(kendall_tau(THETA, m))
            inv["C"][K].append(adjacent_inversions(THETA, m))
            ciw[K].append(float(np.mean(hi - lo)))
            cover[K].append(float(np.mean((THETA >= lo) & (THETA <= hi))))

        # saturazione: set invecchiato da 200 vs static moderno 200 vs CAT 60
        it_leg = rng.choice(legacy.n, 200, replace=False)
        acc_leg = simulate_static(legacy, THETA, it_leg, rng).mean(1)
        sat["tau5_legacy"].append(kendall_tau(THETA[TOP5], acc_leg[TOP5]))
        sat["top1_legacy"].append(float(np.argmax(acc_leg) == np.argmax(THETA)))
        sat["accs_top5_legacy"].append(acc_leg[TOP5].tolist())

        it_mod = rng.choice(bank.n, 200, replace=False)
        acc_mod = simulate_static(bank, THETA, it_mod, rng).mean(1)
        sat["tau5_static200"].append(kendall_tau(THETA[TOP5], acc_mod[TOP5]))
        sat["top1_static200"].append(float(np.argmax(acc_mod) == np.argmax(THETA)))

        m60 = res[60][0]
        sat["tau5_cat60"].append(kendall_tau(THETA[TOP5], m60[TOP5]))
        sat["top1_cat60"].append(float(np.argmax(m60) == np.argmax(THETA)))

    return bank, eap, tau, inv, ciw, cover, sat, np.array(acc240)


# ----------------------------------------------------------------- exp 2

CONTAMINATED = {3: 0.50, 6: 0.25, 9: 0.10, 12: 0.50, 15: 0.25, 17: 0.10}
N_PUB, N_FRESH, N_HP = 200, 200, 20
Z_FLAG = 3.0
MEM_P = 0.98          # prob. che il memorizzatore riproduca la chiave trapelata

def experiment_2(rng):
    pub = make_bank(400, rng, b_lo=-3.0, b_hi=3.0)
    fresh = ItemBank(a=rng.lognormal(0.15, 0.30, 400),
                     b=rng.permutation(pub.b.copy()),
                     c=pub.c.copy())
    eap_pub, eap_fresh = EAPGrid(pub), EAPGrid(fresh)
    true_rank = rank_of(THETA)
    cont_idx = np.array(list(CONTAMINATED.keys()))
    clean_idx = np.array([i for i in range(N_MODELS) if i not in CONTAMINATED])

    res = {"disp_by_f": {0.10: [], 0.25: [], 0.50: []},
           "disp_after_by_f": {0.10: [], 0.25: [], 0.50: []},
           "det_by_f": {0.10: [], 0.25: [], 0.50: []},
           "fpr": [], "z_clean": [], "z_cont": [],
           "tau_statusquo": [], "tau_corrected": [], "tau_cleanworld": []}

    for r in range(R2):
        pub_set = rng.choice(pub.n, N_PUB, replace=False)
        hp_cols = rng.choice(N_PUB, N_HP, replace=False)         # colonne honeypot
        hp_mask = np.zeros(N_PUB, dtype=bool); hp_mask[hp_cols] = True
        fr_set = rng.choice(fresh.n, N_FRESH, replace=False)

        # memorizzazione: f * 200 item del set trapelato, per modello
        mem_cols = np.zeros((N_MODELS, N_PUB), dtype=bool)
        for mi, f in CONTAMINATED.items():
            mem_cols[mi, rng.choice(N_PUB, int(f * N_PUB), replace=False)] = True

        # --- risposte sul set pubblico (con honeypot)
        p_base = p_correct(THETA[:, None], pub.a[pub_set][None, :],
                           pub.b[pub_set][None, :], pub.c[pub_set][None, :])
        # memorizzatore: riproduce la chiave trapelata -> giusta sui normali,
        # SBAGLIATA (piantata) sugli honeypot
        p_eff = np.where(mem_cols & ~hp_mask[None, :], MEM_P, p_base)
        p_eff = np.where(mem_cols & hp_mask[None, :], 1.0 - MEM_P, p_eff)
        U_pub = (rng.random(p_eff.shape) < p_eff).astype(float)

        # indicatore "ha dato la risposta piantata" sugli honeypot
        planted = np.zeros((N_MODELS, N_PUB), dtype=bool)
        rnd = rng.random(p_eff.shape)
        # non-memorizzatori: se sbagliano un honeypot, 1/3 di prob. che l'errore
        # coincida con la risposta piantata (4 opzioni, 3 distrattori)
        planted |= (~mem_cols) & hp_mask[None, :] & (U_pub < 0.5) & (rnd < 1/3)
        planted |= mem_cols & hp_mask[None, :] & (rng.random(p_eff.shape) < MEM_P)
        planted_count = planted[:, hp_mask].sum(1).astype(float)

        # --- risposte fresche e mondo pulito
        U_fresh = simulate_static(fresh, THETA, fr_set, rng)
        U_clean = simulate_static(pub, THETA, pub_set, rng)
        acc_pub, acc_fresh = U_pub.mean(1), U_fresh.mean(1)

        # --- status quo: classifica per accuracy sul set pubblico
        obs_rank = rank_of(acc_pub)
        disp = true_rank - obs_rank
        for mi, f in CONTAMINATED.items():
            res["disp_by_f"][f].append(int(disp[mi]))
        res["tau_statusquo"].append(kendall_tau(THETA, acc_pub))
        res["tau_cleanworld"].append(kendall_tau(THETA, U_clean.mean(1)))

        # --- rilevamento
        m_fresh, *_ = eap_from_static(eap_fresh, fr_set, U_fresh)
        # z honeypot: baseline attesa dalla abilita' stimata sul fresco
        p_hp = p_correct(m_fresh[:, None], pub.a[pub_set][None, hp_mask][0][None, :],
                         pub.b[pub_set][None, hp_mask][0][None, :],
                         pub.c[pub_set][None, hp_mask][0][None, :])
        base = (1.0 - p_hp) / 3.0
        e0, v0 = base.sum(1), (base * (1 - base)).sum(1)
        z_hp = (planted_count - e0) / np.sqrt(v0)
        # z gap pubblico-fresco (offset di popolazione + varianza binomiale)
        gap = acc_pub - acc_fresh
        offset = np.median(gap)
        pbar = np.clip((acc_pub + acc_fresh) / 2, 0.02, 0.98)
        var_g = pbar * (1 - pbar) * (1 / N_PUB + 1 / N_FRESH)
        z_gap = (gap - offset) / np.sqrt(var_g)
        z_comb = (z_hp + z_gap) / np.sqrt(2.0)
        flagged = (z_hp > Z_FLAG) | (z_gap > Z_FLAG) | (z_comb > Z_FLAG)

        for mi, f in CONTAMINATED.items():
            res["det_by_f"][f].append(bool(flagged[mi]))
        res["fpr"].append(float(flagged[clean_idx].mean()))
        res["z_clean"].extend(np.maximum(z_hp, z_gap)[clean_idx].tolist())
        res["z_cont"].extend(np.maximum(z_hp, z_gap)[cont_idx].tolist())

        # --- correzione: flaggati ri-classificati sul fresco
        m_pub, *_ = eap_from_static(eap_pub, pub_set, U_pub)
        corrected = np.where(flagged, m_fresh, m_pub)
        res["tau_corrected"].append(kendall_tau(THETA, corrected))
        disp_after = true_rank - rank_of(corrected)
        for mi, f in CONTAMINATED.items():
            res["disp_after_by_f"][f].append(int(disp_after[mi]))

    return res


# ----------------------------------------------------------------- main

def summarize(x):
    x = np.asarray(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std()),
            "p25": float(np.percentile(x, 25)), "p75": float(np.percentile(x, 75))}

if __name__ == "__main__":
    rng = np.random.default_rng(MASTER_SEED)
    bank, eap, tau, inv, ciw, cover, sat, acc240 = experiment_1(rng)
    res2 = experiment_2(rng)

    accs5 = np.array(sat["accs_top5_legacy"])          # (R1, 5)
    out = {
        "config": {"n_models": N_MODELS, "theta": THETA.tolist(),
                   "budgets": BUDGETS, "R1": R1, "R2": R2,
                   "contaminated": {str(k): v for k, v in CONTAMINATED.items()},
                   "n_honeypot": N_HP, "z_flag": Z_FLAG, "seed": MASTER_SEED},
        "exp1": {
            "tau": {arm: {str(K): summarize(v[K]) for K in BUDGETS} for arm, v in tau.items()},
            "inversions": {arm: {str(K): summarize(v[K]) for K in BUDGETS} for arm, v in inv.items()},
            "ci_width": {str(K): summarize(ciw[K]) for K in BUDGETS},
            "coverage": {str(K): summarize(cover[K]) for K in BUDGETS},
            "saturation": {
                "tau5_legacy": summarize(sat["tau5_legacy"]),
                "tau5_static200": summarize(sat["tau5_static200"]),
                "tau5_cat60": summarize(sat["tau5_cat60"]),
                "top1_legacy": float(np.mean(sat["top1_legacy"])),
                "top1_static200": float(np.mean(sat["top1_static200"])),
                "top1_cat60": float(np.mean(sat["top1_cat60"])),
                "legacy_top5_mean_acc": float(accs5.mean()),
                "legacy_top5_spread_pp": float(100 * (accs5.max(1) - accs5.min(1)).mean()),
            },
            "acc240_rerun_sd_pp": float(100 * np.mean(acc240.std(0))),
            "acc240_adjacent_gap_pp": float(100 * np.mean(np.diff(np.sort(acc240.mean(0))))),
        },
        "exp2": {
            "rank_inflation_by_f": {str(f): summarize(v) for f, v in res2["disp_by_f"].items()},
            "rank_inflation_after_by_f": {str(f): summarize(v) for f, v in res2["disp_after_by_f"].items()},
            "detection_rate_by_f": {str(f): float(np.mean(v)) for f, v in res2["det_by_f"].items()},
            "false_positive_rate": float(np.mean(res2["fpr"])),
            "tau_statusquo": summarize(res2["tau_statusquo"]),
            "tau_corrected": summarize(res2["tau_corrected"]),
            "tau_cleanworld": summarize(res2["tau_cleanworld"]),
        },
    }
    with open("../out/results_static.json", "w") as fh:
        json.dump(out, fh, indent=1)

    np.savez_compressed(
        "../out/arrays_static.npz",
        theta=THETA,
        tau_A=np.array([tau["A"][K] for K in BUDGETS]),
        tau_B=np.array([tau["B"][K] for K in BUDGETS]),
        tau_C=np.array([tau["C"][K] for K in BUDGETS]),
        sat_tau5_legacy=np.array(sat["tau5_legacy"]),
        sat_tau5_static200=np.array(sat["tau5_static200"]),
        sat_tau5_cat60=np.array(sat["tau5_cat60"]),
        sat_accs_top5_legacy=accs5,
        acc240=acc240,
        z_clean=np.array(res2["z_clean"]),
        z_cont=np.array(res2["z_cont"]),
        disp_f10=np.array(res2["disp_by_f"][0.10]),
        disp_f25=np.array(res2["disp_by_f"][0.25]),
        disp_f50=np.array(res2["disp_by_f"][0.50]),
        dispA_f10=np.array(res2["disp_after_by_f"][0.10]),
        dispA_f25=np.array(res2["disp_after_by_f"][0.25]),
        dispA_f50=np.array(res2["disp_after_by_f"][0.50]),
    )
    print("tau per braccio/budget:")
    for arm in "ABC":
        print(" ", arm, {K: round(np.mean(tau[arm][K]), 3) for K in BUDGETS})
    print("coverage CAT:", {K: round(np.mean(cover[K]), 3) for K in BUDGETS})
    print("saturazione tau5: legacy=%.2f static200=%.2f cat60=%.2f | top1: %.2f/%.2f/%.2f | acc_top5=%.3f spread=%.1fpp" % (
        np.mean(sat["tau5_legacy"]), np.mean(sat["tau5_static200"]), np.mean(sat["tau5_cat60"]),
        np.mean(sat["top1_legacy"]), np.mean(sat["top1_static200"]), np.mean(sat["top1_cat60"]),
        accs5.mean(), 100 * (accs5.max(1) - accs5.min(1)).mean()))
    print("exp2 detection:", out["exp2"]["detection_rate_by_f"], "FPR:", round(out["exp2"]["false_positive_rate"], 4))
    print("exp2 inflazione posizioni:", {f: round(np.mean(v), 1) for f, v in res2["disp_by_f"].items()},
          "dopo correzione:", {f: round(np.mean(v), 1) for f, v in res2["disp_after_by_f"].items()})
    print("tau statusquo/corretto/pulito: %.3f / %.3f / %.3f" % (
        np.mean(res2["tau_statusquo"]), np.mean(res2["tau_corrected"]), np.mean(res2["tau_cleanworld"])))
