"""
BenchLab PoC - engine psicometrico.

Componenti:
  - IRT 3PL (three-parameter logistic): P(corretta) = c + (1-c)/(1+exp(-a(theta-b)))
  - Stima bayesiana EAP su griglia (posterior mean + intervallo credibile 95%)
  - CAT (Computerized Adaptive Testing): selezione item per massima informazione
    di Fisher con exposure control "randomesque" (scelta casuale tra i top-k)
  - Bradley-Terry MLE per classifiche da voti pairwise (stile LMArena)

Tutto vettorizzato numpy. Nessuna dipendenza oltre numpy/scipy.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize


# ------------------------------------------------------------------ IRT core

def p_correct(theta, a, b, c):
    """Probabilita' 3PL di risposta corretta. Supporta broadcasting."""
    return c + (1.0 - c) / (1.0 + np.exp(-a * (theta - b)))


def fisher_info(theta, a, b, c):
    """Informazione di Fisher dell'item alla abilita' theta (3PL)."""
    p = p_correct(theta, a, b, c)
    q = 1.0 - p
    return (a ** 2) * (q / np.clip(p, 1e-9, None)) * ((p - c) / (1.0 - c)) ** 2


@dataclass
class ItemBank:
    a: np.ndarray  # discriminativita'
    b: np.ndarray  # difficolta'
    c: np.ndarray  # pseudo-guessing (0.25 = scelta multipla a 4 opzioni)

    @property
    def n(self):
        return len(self.a)


def make_bank(n_items, rng, b_lo=-3.5, b_hi=4.0, a_mu=0.15, a_sd=0.30, c_val=0.25):
    """Banca item con difficolta' uniforme nel range e discriminativita' lognormale."""
    return ItemBank(
        a=rng.lognormal(a_mu, a_sd, n_items),
        b=rng.uniform(b_lo, b_hi, n_items),
        c=np.full(n_items, c_val),
    )


def make_legacy_bank(n_items, rng, mu=-0.3, sd=0.9, b_cap=1.3, c_val=0.25):
    """Banca 'invecchiata': difficolta' concentrata su facile/medio (stile MMLU 2020)."""
    b = np.clip(rng.normal(mu, sd, n_items), -3.0, b_cap)
    return ItemBank(
        a=rng.lognormal(0.15, 0.30, n_items),
        b=b,
        c=np.full(n_items, c_val),
    )


# ------------------------------------------------------- EAP su griglia

class EAPGrid:
    """Griglia di quadratura per stima bayesiana con tabelle precomputate."""

    def __init__(self, bank, lo=-4.5, hi=4.5, n=181, prior_sd=1.8):
        self.grid = np.linspace(lo, hi, n)
        self.dg = self.grid[1] - self.grid[0]
        self.lo = lo
        self.log_prior = -0.5 * (self.grid / prior_sd) ** 2
        G = self.grid[:, None]
        P = p_correct(G, bank.a[None, :], bank.b[None, :], bank.c[None, :])
        P = np.clip(P, 1e-9, 1 - 1e-9)
        self.log_p = np.log(P)          # (n_grid, n_items)
        self.log_q = np.log(1.0 - P)    # (n_grid, n_items)
        self.info = fisher_info(G, bank.a[None, :], bank.b[None, :], bank.c[None, :])

    def posterior_stats(self, loglik):
        """loglik: (..., n_grid). Ritorna (mean, sd, low95, high95)."""
        lp = loglik + self.log_prior
        lp = lp - lp.max(axis=-1, keepdims=True)
        w = np.exp(lp)
        w /= w.sum(axis=-1, keepdims=True)
        mean = (w * self.grid).sum(-1)
        var = (w * (self.grid - mean[..., None]) ** 2).sum(-1)
        cdf = np.cumsum(w, axis=-1)
        lo_i = np.argmax(cdf >= 0.025, axis=-1)
        hi_i = np.argmax(cdf >= 0.975, axis=-1)
        return mean, np.sqrt(var), self.grid[lo_i], self.grid[hi_i]

    def theta_to_idx(self, theta):
        idx = np.rint((theta - self.lo) / self.dg).astype(int)
        return np.clip(idx, 0, len(self.grid) - 1)


# ------------------------------------------------------------- simulazione risposte

def simulate_static(bank, theta_true, item_idx, rng, memorized=None, mem_p=0.98):
    """
    Risposte di E esaminandi sugli stessi K item (benchmark statico).
    theta_true: (E,); item_idx: (K,); memorized: (E, n_items) bool opzionale.
    Ritorna U: (E, K) in {0,1}.
    """
    p = p_correct(theta_true[:, None],
                  bank.a[item_idx][None, :],
                  bank.b[item_idx][None, :],
                  bank.c[item_idx][None, :])
    if memorized is not None:
        p = np.where(memorized[:, item_idx], mem_p, p)
    return (rng.random(p.shape) < p).astype(np.float64)


def eap_from_static(eap, item_idx, U):
    """Stima EAP dalle risposte su un set fisso. U: (E, K)."""
    lp = eap.log_p[:, item_idx]   # (n_grid, K)
    lq = eap.log_q[:, item_idx]
    loglik = U @ lp.T + (1.0 - U) @ lq.T   # (E, n_grid)
    return eap.posterior_stats(loglik)


def run_cat(bank, eap, theta_true, budgets, rng, top_k=8,
            memorized=None, mem_p=0.98):
    """
    CAT vettorizzato su E esaminandi in parallelo.
    budgets: checkpoint crescenti (es. [30, 60, 120, 240]).
    Ritorna: dict budget -> (mean, sd, lo95, hi95), piu' conteggio item somministrati.
    """
    E = len(theta_true)
    N = bank.n
    n_grid = len(eap.grid)
    loglik = np.zeros((E, n_grid))
    administered = np.zeros((E, N), dtype=bool)
    theta_idx = np.full(E, n_grid // 2)
    out = {}
    kmax = max(budgets)
    rows = np.arange(E)

    for step in range(kmax):
        info = eap.info[theta_idx].copy()            # (E, N)
        info[administered] = -np.inf
        # randomesque exposure control: scegli a caso tra i top_k informativi
        part = np.argpartition(-info, top_k, axis=1)[:, :top_k]
        choice = part[rows, rng.integers(0, top_k, E)]
        administered[rows, choice] = True

        p = p_correct(theta_true, bank.a[choice], bank.b[choice], bank.c[choice])
        if memorized is not None:
            p = np.where(memorized[rows, choice], mem_p, p)
        u = rng.random(E) < p

        upd = np.where(u[:, None], eap.log_p[:, choice].T, eap.log_q[:, choice].T)
        loglik += upd

        mean, sd, lo, hi = eap.posterior_stats(loglik)
        theta_idx = eap.theta_to_idx(mean)

        if (step + 1) in budgets:
            out[step + 1] = (mean.copy(), sd.copy(), lo.copy(), hi.copy())

    return out


# ------------------------------------------------------------- Bradley-Terry

def fit_bt(idx_a, idx_b, score_a, n_models, weights=None, reg=0.05, x0=None):
    """
    MLE Bradley-Terry. score_a in {1, 0.5, 0} (vittoria A / pareggio / vittoria B).
    weights: peso per battaglia (difese anti-manipolazione). reg: ridge leggera.
    Ritorna strengths (media zero) sulla scala naturale; Elo = s * 400/ln(10) + 1000.
    """
    if weights is None:
        weights = np.ones(len(idx_a))

    def negloglik(s):
        d = s[idx_a] - s[idx_b]
        # log-verosimiglianza pesata di Bernoulli con esiti frazionari (tie=0.5)
        lp = -np.logaddexp(0.0, -d)   # log sigmoid(d)
        lq = -np.logaddexp(0.0, d)    # log sigmoid(-d)
        ll = weights * (score_a * lp + (1.0 - score_a) * lq)
        return -ll.sum() + reg * np.sum(s ** 2)

    def grad(s):
        d = s[idx_a] - s[idx_b]
        sig = 1.0 / (1.0 + np.exp(-d))
        gcoef = weights * (score_a - sig)
        g = np.zeros(n_models)
        np.add.at(g, idx_a, -gcoef)
        np.add.at(g, idx_b, gcoef)
        return g + 2.0 * reg * s

    s0 = np.zeros(n_models) if x0 is None else x0.copy()
    res = minimize(negloglik, s0, jac=grad, method="L-BFGS-B",
                   options={"maxiter": 500, "ftol": 1e-12})
    s = res.x - res.x.mean()
    return s


def bt_to_elo(s):
    return s * (400.0 / np.log(10.0)) + 1000.0


# ------------------------------------------------------------- metriche

def kendall_tau(x, y):
    from scipy.stats import kendalltau
    return kendalltau(x, y).statistic


def adjacent_inversions(theta_true, theta_hat):
    """Frazione di coppie adiacenti (per rank vero) invertite nella stima."""
    order = np.argsort(theta_true)
    t = theta_hat[order]
    return float(np.mean(np.diff(t) < 0))


def rank_of(values, descending=True):
    """Rank 1 = migliore."""
    order = np.argsort(-values if descending else values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(values) + 1)
    return ranks
