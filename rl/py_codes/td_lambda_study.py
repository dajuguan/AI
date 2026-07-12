"""
Study of lambda: feel the bias-variance trade-off and sweet spot of TD(lambda).

MRP: 19-state random walk (Sutton & Barto, Example 7.1 / Fig 12.6)
  - non-terminal states 1..19, terminal states 0 and 20, start at 10
  - equal-probability left/right step; +1 on entering 20, -1 on entering 0, else 0; gamma = 1
  - true value is analytic: v(i) = (i - 10) / 10, i.e. -0.9 .. +0.9

Figures produced:
  1. lambda-sweet-spot.svg        -- U-shape of RMS vs lambda (both ends bad, middle best)
  2. bias-variance-targets.svg    -- distribution of MC/TD(0)/TD(lambda) learning targets at a fixed state
  3. estimator-bias-variance-vs0.svg -- bias/variance of the trained estimator V_hat(s0), across runs
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Make Chinese render (system has Droid Sans Fallback for CJK): registering the font file explicitly is most robust
import os as _os
from matplotlib import font_manager as _fm
_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if _os.path.exists(_CJK):
    _fm.fontManager.addfont(_CJK)
    _cjk_name = _fm.FontProperties(fname=_CJK).get_name()
    # Per-glyph fallback: Latin/digits use DejaVu (Droid lacks even "0" and "."), CJK falls back to Droid
    plt.rcParams["font.family"] = ["DejaVu Sans", _cjk_name]
plt.rcParams["axes.unicode_minus"] = False

# Output dir: script lives in rl/py_codes/, figures go to rl/imgs/ (relative to the script, correct from any cwd)
IMG_DIR = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          "..", "imgs"))
_os.makedirs(IMG_DIR, exist_ok=True)

N = 19                                    # number of non-terminal states
LEFT, RIGHT = 0, N + 1                    # terminal states 0, 20
START = (N + 1) // 2                      # start state 10
GAMMA = 1.0
STATES = np.arange(1, N + 1)
TRUE_V = (STATES - START) / (START)       # v(i) = (i-10)/10  -> -0.9..0.9


def gen_episode(rng, start=START):
    """Return a trajectory [(s, r, s_next), ...], where every s is non-terminal."""
    s = start
    traj = []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        r = 1.0 if s2 == RIGHT else (-1.0 if s2 == LEFT else 0.0)
        traj.append((s, r, s2))
        s = s2
        if s == LEFT or s == RIGHT:
            return traj


def td_lambda_rms_vs0(episodes, lam, alpha, s0=15):
    """Run online TD(lambda) (accumulating traces) on a batch of episodes.

    Returns (mean per-episode RMS, list of V_hat(s0) after each episode).
    """
    V = np.zeros(N + 2)                    # V[0], V[20] stay 0
    rms = []
    V_S0 = []
    # Large alpha + large lambda can diverge; caught by the isfinite check below.
    # Divergence is expected here, so locally silence its overflow/NaN warnings.
    with np.errstate(over="ignore", invalid="ignore"):
        for traj in episodes:
            E = np.zeros(N + 2)
            for (s, r, s2) in traj:
                delta = r + GAMMA * V[s2] - V[s]
                E[s] += 1.0
                V += alpha * delta * E
                V[LEFT] = 0.0
                V[RIGHT] = 0.0
                E *= GAMMA * lam
            if not np.all(np.isfinite(V)):     # diverged: flag as large error, dropped when taking best alpha
                return 1e3, None
            rms.append(np.sqrt(np.mean((V[1:N + 1] - TRUE_V) ** 2)))
            V_S0.append(V[s0])                  # record the value estimate at s0 for bias/variance analysis
    return np.mean(rms), V_S0


def lambda_return_from(traj, V, lam):
    """Given a trajectory from some start state and a value table V, compute the lambda-return of that start (with residual term)."""
    rewards = [r for (_, r, _) in traj]
    nexts = [s2 for (_, _, s2) in traj]
    T = len(traj)                          # steps taken to termination
    # n-step returns G^(n), n = 1..T (for n>=T it equals the full return)
    Gn = np.zeros(T + 1)
    G_full = sum((GAMMA ** k) * rewards[k] for k in range(T))
    acc = 0.0
    for n in range(1, T + 1):
        acc += (GAMMA ** (n - 1)) * rewards[n - 1]
        Gn[n] = acc + (GAMMA ** n) * V[nexts[n - 1]]   # with V[terminal]=0 this auto-equals G_full
    if lam == 1.0:
        return G_full
    gl = 0.0
    for n in range(1, T):
        gl += (lam ** (n - 1)) * Gn[n]
    gl *= (1 - lam)
    gl += (lam ** (T - 1)) * G_full        # residual term: after termination all weight goes to the full return
    return gl


# ---------------------------------------------------------------- Experiment 1: sweet spot
def experiment_sweetspot(runs=100, eps_per_run=10, seed=0):
    lambdas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0])
    alphas = np.round(np.arange(0.02, 0.62, 0.04), 3)
    err = np.zeros((len(lambdas), len(alphas)))   # accumulate RMS
    rng = np.random.default_rng(seed)
    for _ in range(runs):
        episodes = [gen_episode(rng) for _ in range(eps_per_run)]  # shared random numbers
        for i, lam in enumerate(lambdas):
            for j, al in enumerate(alphas):
                err[i, j] += td_lambda_rms_vs0(episodes, lam, al)[0]
    err /= runs
    return lambdas, alphas, err


# ---------------------------------------------------------------- Experiment 2: bias/variance of the target
def experiment_targets(n_samples=20000, s0=15, seed=1):
    """Sweep lambda; for targets starting at s0, collect their mean (-> bias) and std (-> variance)."""
    rng = np.random.default_rng(seed)
    trajs = [gen_episode(rng, start=s0) for _ in range(n_samples)]
    V_true = np.zeros(N + 2)
    V_true[1:N + 1] = TRUE_V
    V_zero = np.zeros(N + 2)
    lams = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    out = {}
    for tag, V in [("true", V_true), ("zero", V_zero)]:
        means, stds = [], []
        for lam in lams:
            g = np.array([lambda_return_from(t, V, float(lam)) for t in trajs])
            means.append(g.mean()); stds.append(g.std())
        out[tag] = (np.array(means), np.array(stds))
    return s0, lams, out


# ---------------------------------------------------------------- Experiment 3: bias/variance of the estimator V_hat(s0) (across runs)
def experiment_estimator_vs0(runs=50, eps_per_run=300, s0=15, alpha=0.05, seed=2):
    """Across independent runs, record V_hat(s0) after each episode.

    The proper estimator variance Var(V_hat(s0)) is the spread of the FINAL
    V_hat(s0) ACROSS independent training runs (different random datasets),
    NOT the within-run spread of V_hat(s0) over episodes. Likewise the bias is
    (mean of the final V_hat(s0) across runs) - true value.
    Returns curves[lam] of shape (runs, eps_per_run).
    """
    lams = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0])
    curves = {float(l): [] for l in lams}
    rng = np.random.default_rng(seed)
    for _ in range(runs):
        episodes = [gen_episode(rng) for _ in range(eps_per_run)]   # from center, shared across lambdas
        for lam in lams:
            _, v_s0 = td_lambda_rms_vs0(episodes, float(lam), alpha, s0=s0)
            if v_s0 is None:                      # diverged (should not happen at small alpha)
                continue
            curves[float(lam)].append(v_s0)
    for k in curves:
        curves[k] = np.array(curves[k])           # (runs, eps_per_run)
    return s0, lams, alpha, curves


# ---------------------------------------------------------------- plotting
def plot_estimator_vs0(s0, lams, alpha, curves):
    truev = (s0 - START) / START
    n_runs, n_eps = curves[float(lams[0])].shape
    eps_axis = np.arange(1, n_eps + 1)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))

    # (a) convergence of V_hat(s0): mean line +/- 1 std band across runs
    show = [0.0, 0.4, 0.6, 0.8, 1.0]
    # high-contrast: lambda=0 black, then blue -> green -> amber -> red as lambda (and variance) grow
    colors = {0.0: "#000000", 0.4: "#3b7dd8", 0.6: "#17a77e",
              0.8: "#e6a700", 1.0: "#d1495b"}
    # band colors match the lines, except lambda=0's black line uses a light-grey band
    band_colors = {**colors, 0.0: "#9e9e9e"}
    # draw every std band first, then all mean lines on top (so later bands can't hide earlier lines)
    for lam in show:
        arr = curves[float(lam)]                  # (runs, eps)
        m, sd = arr.mean(axis=0), arr.std(axis=0)
        axL.fill_between(eps_axis, m - sd, m + sd, color=band_colors[lam], alpha=0.12)
    for lam in show:
        m = curves[float(lam)].mean(axis=0)
        axL.plot(eps_axis, m, "-", lw=2.0, color=colors[lam], label=f"λ={lam:g}")
    axL.axhline(truev, color="k", ls="--", lw=1.2)
    axL.text(eps_axis[0], truev + 0.03, f"true value = {truev:+.1f}", fontsize=9)
    axL.set_xlabel("episode"); axL.set_ylabel(fr"$\hat{{V}}(s_0)$ over {n_runs} runs")
    axL.set_title(r"(a) Convergence of $\hat{V}(s_0)$ (mean line, $\pm$1 std band)")
    axL.legend(fontsize=8, ncol=2); axL.grid(alpha=0.3)

    # (b) end of training: mean (-> bias) and std (-> estimator variance) across runs
    finals_mean = np.array([curves[float(l)][:, -1].mean() for l in lams])
    finals_std = np.array([curves[float(l)][:, -1].std() for l in lams])
    axR.axhline(truev, color="k", ls="--", lw=1.2)
    axR.text(0.02, truev + 0.03, f"true value = {truev:+.1f}", fontsize=9)
    axR.errorbar(lams, finals_mean, yerr=finals_std, fmt="o-", color="#d1495b",
                 ecolor="#3b7dd8", elinewidth=2, capsize=5, ms=6, lw=1.8,
                 label=r"final $\hat{V}(s_0)$: mean $\pm$ 1 std")
    axR.annotate("λ=0\npure TD(0)", (lams[0], finals_mean[0]),
                 textcoords="offset points", xytext=(8, -30), fontsize=8, color="#555")
    axR.annotate("λ=1\npure MC", (lams[-1], finals_mean[-1]),
                 textcoords="offset points", xytext=(-36, 12), fontsize=8, color="#555")
    axR.set_xlabel("λ"); axR.set_ylabel(r"final $\hat{V}(s_0)$ across runs")
    axR.set_title("(b) bias (point vs true) & estimator variance (bar)")
    axR.legend(loc="lower right", fontsize=8); axR.grid(alpha=0.3); axR.set_xlim(-0.05, 1.08)

    fig.suptitle(fr"Estimator bias vs variance of $\hat{{V}}(s_0)$ "
                 fr"(state {s0}, $\alpha$={alpha}, across {n_runs} runs)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_os.path.join(IMG_DIR, "estimator-bias-variance-vs0.svg"))
    print("saved estimator-bias-variance-vs0.svg")
    print("  final mean (-> bias vs %.1f):" % truev, np.round(finals_mean, 3))
    print("  final std  (-> estimator var):", np.round(finals_std, 3))


def plot_sweetspot(lambdas, alphas, err):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    show = [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(show)))
    for c, lam in zip(cmap, show):
        i = int(np.argmin(np.abs(lambdas - lam)))
        axL.plot(alphas, err[i], "-o", ms=3, color=c, label=f"λ={lam:g}")
    axL.set_xlabel("step size α"); axL.set_ylabel("mean RMS error (first 10 episodes)")
    axL.set_title("(a) RMS vs α for each λ"); axL.set_ylim(0.1, 0.55)
    axL.legend(fontsize=8, ncol=2); axL.grid(alpha=0.3)

    best = err.min(axis=1)                 # best alpha for each lambda
    axR.plot(lambdas, best, "-o", color="#d1495b", lw=2, ms=5)
    k = int(np.argmin(best))
    axR.scatter([lambdas[k]], [best[k]], s=140, facecolors="none",
                edgecolors="#2e7d32", lw=2, zorder=5)
    axR.annotate(f"sweet spot\nλ≈{lambdas[k]:g}", (lambdas[k], best[k]),
                 textcoords="offset points", xytext=(10, 22), fontsize=9,
                 color="#2e7d32",
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    axR.annotate("λ=0\npure TD(0)\nbias end", (lambdas[0], best[0]),
                 textcoords="offset points", xytext=(12, 4), fontsize=8, color="#555")
    axR.annotate("λ=1\npure MC\nvariance end", (lambdas[-1], best[-1]),
                 textcoords="offset points", xytext=(-30, 6), fontsize=8, color="#555")
    axR.set_xlabel("λ"); axR.set_ylabel("mean RMS error at best α")
    axR.set_title("(b) performance vs λ: both ends worse, sweet spot in the middle"); axR.grid(alpha=0.3)
    fig.suptitle("Study of λ (19-state random walk):Should We Bootstrap?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(_os.path.join(IMG_DIR, "lambda-sweet-spot.svg"))
    print("saved lambda-sweet-spot.svg; best λ =", lambdas[k], "RMS =", round(best[k], 4))


def plot_targets(s0, lams, out):
    truev = (s0 - START) / START
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    titles = {"true": "(a) V = true value: all unbiased (mean on true line), variance ↑ as λ ↑",
              "zero": "(b) V = 0 (untrained): λ=0 biased → λ=1 unbiased, but variance ↑ too"}
    for ax, tag in zip(axes, ["true", "zero"]):
        means, stds = out[tag]
        ax.axhline(truev, color="k", ls="--", lw=1.2, zorder=1)
        ax.text(0.02, truev + 0.03, f"true value = {truev:+.1f}", fontsize=9)
        # error bars: point=mean (bias), half-length=std (variance)
        ax.errorbar(lams, means, yerr=stds, fmt="o-", color="#d1495b",
                    ecolor="#3b7dd8", elinewidth=2, capsize=5, ms=6, lw=1.8,
                    zorder=3, label="mean ± 1 std")
        # annotate the two ends
        ax.annotate("λ=0\npure TD(0)", (lams[0], means[0]),
                    textcoords="offset points", xytext=(8, -28), fontsize=8, color="#555")
        ax.annotate("λ=1\npure MC", (lams[-1], means[-1]),
                    textcoords="offset points", xytext=(-34, 10), fontsize=8, color="#555")
        ax.set_title(titles[tag], fontsize=10)
        ax.set_xlabel("λ"); ax.grid(alpha=0.3); ax.set_xlim(-0.05, 1.08)
    axes[0].set_ylabel(f"learning target from state {s0}\n(point=mean→bias, bar=std→variance)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"target bias vs variance (random walk state {s0}, true value {truev:+.1f})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_os.path.join(IMG_DIR, "bias-variance-targets.svg"))
    print("saved bias-variance-targets.svg; V=true stds:", np.round(out['true'][1], 3),
          "| V=0 means:", np.round(out['zero'][0], 3))


def plot_walk_diagram():
    """Draw the MRP structure of the 19-state random walk (states colored by true value)."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(12, 3.2))
    # non-terminal states: dots colored by true value v(i)
    sc = ax.scatter(STATES, np.zeros(N), c=TRUE_V, cmap="coolwarm",
                    vmin=-1, vmax=1, s=520, edgecolors="k", linewidths=1.2, zorder=3)
    for i in STATES:
        ax.text(i, 0, str(i), ha="center", va="center", fontsize=8, zorder=4)
    # terminal states: squares
    for x, r, col in [(LEFT, "-1", "#3b4cc0"), (RIGHT, "+1", "#b40426")]:
        ax.add_patch(Rectangle((x - 0.32, -0.32), 0.64, 0.64, facecolor=col,
                               edgecolor="k", alpha=0.85, zorder=3))
        ax.text(x, 0, "T", ha="center", va="center", color="w", fontsize=11,
                zorder=4)
        ax.text(x, -0.62, f"terminal\nr={r}", ha="center", va="top", fontsize=9)
    # start-state annotation
    ax.annotate("start", (START, 0), textcoords="offset points", xytext=(0, 34),
                ha="center", fontsize=10, color="#2e7d32",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5))
    # representative left/right transition arrows (illustrated near state 5, each step +-1 with prob 0.5)
    for a, b, dy in [(5, 6, 0.22), (6, 5, -0.22)]:
        ax.add_patch(FancyArrowPatch((a, dy * 0.6), (b, dy * 0.6),
                     connectionstyle=f"arc3,rad={0.35 if dy>0 else -0.35}",
                     arrowstyle="-|>", mutation_scale=13, color="#444", zorder=2))
    ax.text(5.5, 0.55, "each step ±1 (prob 0.5)", ha="center", fontsize=9, color="#444")
    ax.text(10, -1.15, "middle transitions r = 0;  +1 on right end, −1 on left end;  γ=1;  true value v(i)=(i−10)/10",
            ha="center", fontsize=9, color="#333")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("true value v(i)", fontsize=9)
    ax.set_xlim(-1.2, 21.2); ax.set_ylim(-1.4, 1.0); ax.axis("off")
    ax.set_title("19-state random walk (states colored by true value)", fontsize=12)
    fig.tight_layout()
    fig.savefig(_os.path.join(IMG_DIR, "random-walk-diagram.svg"))
    print("saved random-walk-diagram.svg")


if __name__ == "__main__":
    import sys
    if "--smoke" in sys.argv:
        import time
        t0 = time.time()
        L, A, err = experiment_sweetspot(runs=5)
        print("smoke sweetspot", round(time.time() - t0, 2), "s; err shape", err.shape)
        t0 = time.time()
        s0, lams, out = experiment_targets(n_samples=2000)
        print("smoke targets", round(time.time() - t0, 2), "s")
        t0 = time.time()
        s0, lams, alpha, curves = experiment_estimator_vs0(runs=5, eps_per_run=30)
        print("smoke estimator", round(time.time() - t0, 2), "s")
    else:
        plot_walk_diagram()
        L, A, err = experiment_sweetspot(runs=100)
        plot_sweetspot(L, A, err)
        s0, lams, out = experiment_targets(n_samples=20000)
        plot_targets(s0, lams, out)
        s0, lams, alpha, curves = experiment_estimator_vs0()
        plot_estimator_vs0(s0, lams, alpha, curves)
