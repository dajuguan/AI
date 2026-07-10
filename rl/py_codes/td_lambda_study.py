"""
Study of lambda: 感受 TD(lambda) 的 bias-variance 权衡与 sweet spot。

MRP: 19-state random walk (Sutton & Barto, Example 7.1 / Fig 12.6)
  - 非终止状态 1..19, 终止态 0 与 20, 起点 10
  - 等概率左右走; 走入 20 得 +1, 走入 0 得 -1, 其余 0; gamma = 1
  - 真值可解析: v(i) = (i - 10) / 10, 即 -0.9 .. +0.9

产出两张图:
  1. lambda-sweet-spot.svg    —— RMS 对 lambda 的 U 形 (两端差、中间最优)
  2. bias-variance-targets.svg —— 固定状态处, MC/TD(0)/TD(lambda) 学习目标的分布
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 让中文正常渲染(系统里有 Droid Sans Fallback 覆盖 CJK):显式注册字体文件最稳妥
import os as _os
from matplotlib import font_manager as _fm
_CJK = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
if _os.path.exists(_CJK):
    _fm.fontManager.addfont(_CJK)
    _cjk_name = _fm.FontProperties(fname=_CJK).get_name()
    # 逐字回退:拉丁/数字用 DejaVu(Droid 连 "0"、"." 都没有),中文回退到 Droid
    plt.rcParams["font.family"] = ["DejaVu Sans", _cjk_name]
plt.rcParams["axes.unicode_minus"] = False

# 图片输出目录: 脚本在 rl/py_codes/, 图放到 rl/imgs/ (相对脚本自身, 从任何 cwd 运行都对)
IMG_DIR = _os.path.normpath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                          "..", "imgs"))
_os.makedirs(IMG_DIR, exist_ok=True)

N = 19                                    # 非终止状态数
LEFT, RIGHT = 0, N + 1                    # 终止态 0, 20
START = (N + 1) // 2                      # 起点 10
GAMMA = 1.0
STATES = np.arange(1, N + 1)
TRUE_V = (STATES - START) / (START)       # v(i) = (i-10)/10  -> -0.9..0.9


def gen_episode(rng, start=START):
    """返回轨迹 [(s, r, s_next), ...], s 均为非终止态。"""
    s = start
    traj = []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1)
        r = 1.0 if s2 == RIGHT else (-1.0 if s2 == LEFT else 0.0)
        traj.append((s, r, s2))
        s = s2
        if s == LEFT or s == RIGHT:
            return traj


def td_lambda_rms(episodes, lam, alpha):
    """在给定的一批 episode 上跑在线 TD(lambda)(累积迹), 返回逐 episode 的平均 RMS。"""
    V = np.zeros(N + 2)                    # V[0], V[20] 恒为 0
    rms = []
    for traj in episodes:
        E = np.zeros(N + 2)
        for (s, r, s2) in traj:
            delta = r + GAMMA * V[s2] - V[s]
            E[s] += 1.0
            V += alpha * delta * E
            V[LEFT] = 0.0
            V[RIGHT] = 0.0
            E *= GAMMA * lam
        if not np.all(np.isfinite(V)):     # 发散(大 α + 大 λ):记为大误差, 由取最优 α 剔除
            return 1e3
        rms.append(np.sqrt(np.mean((V[1:N + 1] - TRUE_V) ** 2)))
    return np.mean(rms)


def lambda_return_from(traj, V, lam):
    """给定从某状态开始的轨迹与价值表 V, 计算该起点的 lambda-return(带残差项)。"""
    rewards = [r for (_, r, _) in traj]
    nexts = [s2 for (_, _, s2) in traj]
    T = len(traj)                          # 到终止用了 T 步
    # 各 n-step 回报 G^(n), n = 1..T (n>=T 时即完整回报)
    Gn = np.zeros(T + 1)
    G_full = sum((GAMMA ** k) * rewards[k] for k in range(T))
    acc = 0.0
    for n in range(1, T + 1):
        acc += (GAMMA ** (n - 1)) * rewards[n - 1]
        Gn[n] = acc + (GAMMA ** n) * V[nexts[n - 1]]   # V[终止]=0 时自动等于 G_full
    if lam == 1.0:
        return G_full
    gl = 0.0
    for n in range(1, T):
        gl += (lam ** (n - 1)) * Gn[n]
    gl *= (1 - lam)
    gl += (lam ** (T - 1)) * G_full        # 残差项: 终止后权重全归完整回报
    return gl


# ---------------------------------------------------------------- 实验 1: sweet spot
def experiment_sweetspot(runs=100, eps_per_run=10, seed=0):
    lambdas = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.975, 0.99, 1.0])
    alphas = np.round(np.arange(0.02, 0.62, 0.04), 3)
    err = np.zeros((len(lambdas), len(alphas)))   # 累加 RMS
    rng = np.random.default_rng(seed)
    for _ in range(runs):
        episodes = [gen_episode(rng) for _ in range(eps_per_run)]  # 共用随机数
        for i, lam in enumerate(lambdas):
            for j, al in enumerate(alphas):
                err[i, j] += td_lambda_rms(episodes, lam, al)
    err /= runs
    return lambdas, alphas, err


# ---------------------------------------------------------------- 实验 2: 目标的偏差/方差
def experiment_targets(n_samples=20000, s0=15, seed=1):
    """扫一串 lambda, 统计从 s0 出发的学习目标的均值(→偏差)与标准差(→方差)。"""
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


# ---------------------------------------------------------------- 画图
def plot_sweetspot(lambdas, alphas, err):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.2))
    show = [0.0, 0.4, 0.8, 0.9, 0.95, 1.0]
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(show)))
    for c, lam in zip(cmap, show):
        i = int(np.argmin(np.abs(lambdas - lam)))
        axL.plot(alphas, err[i], "-o", ms=3, color=c, label=f"λ={lam:g}")
    axL.set_xlabel("步长 α"); axL.set_ylabel("平均 RMS 误差(前 10 episode)")
    axL.set_title("(a) 每个 λ 对 α 的曲线"); axL.set_ylim(0.1, 0.55)
    axL.legend(fontsize=8, ncol=2); axL.grid(alpha=0.3)

    best = err.min(axis=1)                 # 每个 λ 取最优 α
    axR.plot(lambdas, best, "-o", color="#d1495b", lw=2, ms=5)
    k = int(np.argmin(best))
    axR.scatter([lambdas[k]], [best[k]], s=140, facecolors="none",
                edgecolors="#2e7d32", lw=2, zorder=5)
    axR.annotate(f"sweet spot\nλ≈{lambdas[k]:g}", (lambdas[k], best[k]),
                 textcoords="offset points", xytext=(10, 22), fontsize=9,
                 color="#2e7d32",
                 arrowprops=dict(arrowstyle="->", color="#2e7d32"))
    axR.annotate("λ=0\n纯 TD(0)\n偏差端", (lambdas[0], best[0]),
                 textcoords="offset points", xytext=(12, 4), fontsize=8, color="#555")
    axR.annotate("λ=1\n纯 MC\n方差端", (lambdas[-1], best[-1]),
                 textcoords="offset points", xytext=(-30, 6), fontsize=8, color="#555")
    axR.set_xlabel("λ"); axR.set_ylabel("最优 α 下的平均 RMS 误差")
    axR.set_title("(b) 性能对 λ:两端都差,中间有 sweet spot"); axR.grid(alpha=0.3)
    fig.suptitle("Study of λ (19-state random walk):Should We Bootstrap?", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(_os.path.join(IMG_DIR, "lambda-sweet-spot.svg"))
    print("saved lambda-sweet-spot.svg; best λ =", lambdas[k], "RMS =", round(best[k], 4))


def plot_targets(s0, lams, out):
    truev = (s0 - START) / START
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    titles = {"true": "(a) V = 真值:三者都无偏(均值贴真值线),λ↑ 方差↑",
              "zero": "(b) V = 0 未训练:λ=0 偏差大→λ=1 无偏,但方差同步↑"}
    for ax, tag in zip(axes, ["true", "zero"]):
        means, stds = out[tag]
        ax.axhline(truev, color="k", ls="--", lw=1.2, zorder=1)
        ax.text(0.02, truev + 0.03, f"真值 = {truev:+.1f}", fontsize=9)
        # 误差棒: 点=均值(偏差), 半长=标准差(方差)
        ax.errorbar(lams, means, yerr=stds, fmt="o-", color="#d1495b",
                    ecolor="#3b7dd8", elinewidth=2, capsize=5, ms=6, lw=1.8,
                    zorder=3, label="均值 ± 1 标准差")
        # 标注两端
        ax.annotate("λ=0\n纯 TD(0)", (lams[0], means[0]),
                    textcoords="offset points", xytext=(8, -28), fontsize=8, color="#555")
        ax.annotate("λ=1\n纯 MC", (lams[-1], means[-1]),
                    textcoords="offset points", xytext=(-34, 10), fontsize=8, color="#555")
        ax.set_title(titles[tag], fontsize=10)
        ax.set_xlabel("λ"); ax.grid(alpha=0.3); ax.set_xlim(-0.05, 1.08)
    axes[0].set_ylabel(f"从状态 {s0} 出发的学习目标\n(点=均值→偏差,棒长=标准差→方差)")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle(f"学习目标的偏差 vs 方差(random walk 状态 {s0},真值 {truev:+.1f})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(_os.path.join(IMG_DIR, "bias-variance-targets.svg"))
    print("saved bias-variance-targets.svg; V=true stds:", np.round(out['true'][1], 3),
          "| V=0 means:", np.round(out['zero'][0], 3))


def plot_walk_diagram():
    """画出 19-state random walk 的 MRP 结构图(状态按真值上色)。"""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(12, 3.2))
    # 非终止状态: 圆点, 颜色 = 真值 v(i)
    sc = ax.scatter(STATES, np.zeros(N), c=TRUE_V, cmap="coolwarm",
                    vmin=-1, vmax=1, s=520, edgecolors="k", linewidths=1.2, zorder=3)
    for i in STATES:
        ax.text(i, 0, str(i), ha="center", va="center", fontsize=8, zorder=4)
    # 终止态: 方块
    for x, r, col in [(LEFT, "-1", "#3b4cc0"), (RIGHT, "+1", "#b40426")]:
        ax.add_patch(Rectangle((x - 0.32, -0.32), 0.64, 0.64, facecolor=col,
                               edgecolor="k", alpha=0.85, zorder=3))
        ax.text(x, 0, "T", ha="center", va="center", color="w", fontsize=10,
                fontweight="bold", zorder=4)
        ax.text(x, -0.62, f"终止\nr={r}", ha="center", va="top", fontsize=9)
    # 起点标注
    ax.annotate("起点", (START, 0), textcoords="offset points", xytext=(0, 34),
                ha="center", fontsize=10, color="#2e7d32", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.5))
    # 代表性的左右转移箭头(在状态 5 附近示意, 每步 ±1 各 0.5)
    for a, b, dy in [(5, 6, 0.22), (6, 5, -0.22)]:
        ax.add_patch(FancyArrowPatch((a, dy * 0.6), (b, dy * 0.6),
                     connectionstyle=f"arc3,rad={0.35 if dy>0 else -0.35}",
                     arrowstyle="-|>", mutation_scale=13, color="#444", zorder=2))
    ax.text(5.5, 0.55, "每步等概率 ±1\n(各 0.5)", ha="center", fontsize=9, color="#444")
    ax.text(10, -1.15, "中间转移 r = 0;走入右端 +1、左端 −1;γ=1;真值 v(i)=(i−10)/10",
            ha="center", fontsize=9, color="#333")
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("真值 v(i)", fontsize=9)
    ax.set_xlim(-1.2, 21.2); ax.set_ylim(-1.4, 1.0); ax.axis("off")
    ax.set_title("19-state random walk(状态按真值上色)", fontsize=12)
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
    else:
        plot_walk_diagram()
        L, A, err = experiment_sweetspot(runs=100)
        plot_sweetspot(L, A, err)
        s0, lams, out = experiment_targets(n_samples=20000)
        plot_targets(s0, lams, out)
