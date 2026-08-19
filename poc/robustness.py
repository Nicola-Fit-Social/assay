"""
Verifica di robustezza: ripete i risultati chiave su 5 semi indipendenti e
controlla che le conclusioni non dipendano dal seme.

Conclusioni sotto test:
  1. CAT-60 eguaglia o supera lo static-240 valutato ad accuratezza classica
     (lo status quo dei benchmark) -> efficienza ~4x. In ablazione si riporta
     anche lo static-240 con statistica IRT-EAP: con 240 item ben calibrati e
     la statistica giusta CAT-60 e' comparabile, non superiore — come
     dichiarato nel paper (il guadagno e' l'efficienza, non la magia).
  2. Coverage degli intervalli CAT ~ 0.95 (calibrazione onesta)
  3. Correzione contaminazione: tau_corretto >> tau_statusquo, avvicina il mondo pulito
  4. Rilevamento a f=0.5 alto e FPR basso
"""
import numpy as np
from engine import (make_bank, EAPGrid, simulate_static, eap_from_static,
                    run_cat, kendall_tau, rank_of, ItemBank, p_correct)

N_MODELS = 20
THETA = np.linspace(-2.2, 3.0, N_MODELS)
R = 120
CONT = {3: 0.50, 6: 0.25, 9: 0.10, 12: 0.50, 15: 0.25, 17: 0.10}

def one_seed(seed):
    rng = np.random.default_rng(seed)
    bank = make_bank(600, rng)
    eap = EAPGrid(bank)
    tau_acc240, tau_eap240, tau_cat60, cov60 = [], [], [], []
    for _ in range(R):
        items = rng.choice(bank.n, 240, replace=False)
        U = simulate_static(bank, THETA, items, rng)
        tau_acc240.append(kendall_tau(THETA, U.mean(1)))   # status quo: accuratezza
        m, *_ = eap_from_static(eap, items, U)
        tau_eap240.append(kendall_tau(THETA, m))           # ablazione: stessi item, IRT
        res = run_cat(bank, eap, THETA, [60], rng)
        m60, sd, lo, hi = res[60]
        tau_cat60.append(kendall_tau(THETA, m60))
        cov60.append(np.mean((THETA >= lo) & (THETA <= hi)))

    # contaminazione (versione compatta con honeypot + fresco)
    pub = make_bank(400, rng, b_lo=-3.0, b_hi=3.0)
    fresh = ItemBank(rng.lognormal(0.15, 0.30, 400), rng.permutation(pub.b.copy()), pub.c.copy())
    eap_pub, eap_fr = EAPGrid(pub), EAPGrid(fresh)
    true_rank = rank_of(THETA)
    tq, tc, det50, fpr = [], [], [], []
    clean_idx = [i for i in range(N_MODELS) if i not in CONT]
    for _ in range(R):
        ps = rng.choice(pub.n, 200, replace=False)
        hp = np.zeros(200, bool); hp[rng.choice(200, 20, replace=False)] = True
        fs = rng.choice(fresh.n, 200, replace=False)
        mem = np.zeros((N_MODELS, 200), bool)
        for mi, f in CONT.items():
            mem[mi, rng.choice(200, int(f*200), replace=False)] = True
        pb = p_correct(THETA[:, None], pub.a[ps][None], pub.b[ps][None], pub.c[ps][None])
        pe = np.where(mem & ~hp[None], 0.98, pb)
        pe = np.where(mem & hp[None], 0.02, pe)
        Up = (rng.random(pe.shape) < pe).astype(float)
        planted = np.zeros((N_MODELS, 200), bool)
        planted |= (~mem) & hp[None] & (Up < .5) & (rng.random(pe.shape) < 1/3)
        planted |= mem & hp[None] & (rng.random(pe.shape) < 0.98)
        pc = planted[:, hp].sum(1).astype(float)
        Uf = simulate_static(fresh, THETA, fs, rng)
        accp, accf = Up.mean(1), Uf.mean(1)
        mf, *_ = eap_from_static(eap_fr, fs, Uf)
        php = p_correct(mf[:, None], pub.a[ps][None, hp][0][None], pub.b[ps][None, hp][0][None], pub.c[ps][None, hp][0][None])
        base = (1-php)/3; e0 = base.sum(1); v0 = (base*(1-base)).sum(1)
        zhp = (pc - e0)/np.sqrt(v0)
        gap = accp - accf; off = np.median(gap)
        pbar = np.clip((accp+accf)/2, .02, .98); vg = pbar*(1-pbar)*(1/200+1/200)
        zg = (gap-off)/np.sqrt(vg)
        fl = (zhp > 3) | (zg > 3) | ((zhp+zg)/np.sqrt(2) > 3)
        tq.append(kendall_tau(THETA, accp))
        mp, *_ = eap_from_static(eap_pub, ps, Up)
        tc.append(kendall_tau(THETA, np.where(fl, mf, mp)))
        det50.append(np.mean([fl[mi] for mi, f in CONT.items() if f == 0.5]))
        fpr.append(np.mean(fl[clean_idx]))
    return (np.mean(tau_acc240), np.mean(tau_eap240), np.mean(tau_cat60), np.mean(cov60),
            np.mean(tq), np.mean(tc), np.mean(det50), np.mean(fpr))

if __name__ == "__main__":
    print("seed | tau_acc240 tau_eap240 tau_cat60 cov60 | tau_SQ tau_corr det50  FPR")
    rows = []
    for s in [1, 2, 3, 4, 5]:
        r = one_seed(1000 + s)
        rows.append(r)
        print("%4d | %.3f      %.3f      %.3f     %.3f | %.3f  %.3f   %.3f  %.3f" % (s, *r))
    rows = np.array(rows)
    print("\nMedia +/- sd su 5 semi:")
    labels = ["tau_acc240 (status quo)", "tau_eap240 (ablazione)", "tau_cat60", "cov60",
              "tau_statusquo", "tau_corrected", "det@f=.5", "FPR"]
    for i, lab in enumerate(labels):
        print("  %-24s %.3f +/- %.3f" % (lab, rows[:, i].mean(), rows[:, i].std()))
    print("\nCAT-60 >= static-240 ad accuratezza classica in tutti i semi:",
          bool(np.all(rows[:, 2] >= rows[:, 0])))
    print("Ablazione (non un claim): CAT-60 vs EAP-240 comparabili, delta medio %.3f"
          % float(np.mean(rows[:, 2] - rows[:, 1])))
    print("Correzione > status quo in tutti i semi:", bool(np.all(rows[:, 5] > rows[:, 4])))
    print("Coverage in [0.93, 0.97] in tutti i semi:", bool(np.all((rows[:, 3] >= 0.93) & (rows[:, 3] <= 0.97))))
    print("FPR < 2%% in tutti i semi:", bool(np.all(rows[:, 7] < 0.02)))
