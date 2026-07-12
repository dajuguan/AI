# Bias, Variance, and the TD(λ) Sweet Spot — A Hands-On Guide

**TL;DR**
- Revisit what "variance" in RL actually means (the most commonly misunderstood term), and use experiments to pull apart **two kinds of variance**: the variance of the learning target (the *root cause*) and the variance of the value-function estimator (the *consequence*).
- Sweep TD(λ)'s search depth λ on a 19-state random walk, and feel first-hand how search depth affects **convergence speed, variance, and bias**.

## Motivation

In [Lecture 4 of his RL course](https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf), David Silver explains that Monte Carlo (MC) and Temporal Difference (TD) have different variance and bias: **MC has high variance but no bias, while TD has larger bias but low variance**. To illustrate variance he uses the Driving Home Example — but the first time you see it, it is genuinely confusing:

![Driving Home Example](https://cdn.jsdelivr.net/gh/pochenai/AI@main/rl/imgs/driving_home_image.png)

Three things tripped me up:

1. The figure actually **shows only one episode** (one day's commute); a single trajectory gives one number per state — **a single sample simply cannot yield a variance**. Variance needs a bunch of samples before "spread" means anything.
2. Even more awkward: once a value function is **trained**, feeding it a state $s$ returns a **deterministic value** — so where is the variance?
3. And a more basic one: is variance **a single number for the whole value function**, or **one per state**? I kept asking "what is the variance of this $V$?" and went in circles for a while.

Getting past these comes down to being clear about **what the variance is attached to**. First, kill confusion #3: **variance is per-state** — each $s$ has its own target distribution and its own $\hat V(s)$, so each has its own variance; there is no such thing as "one variance for the whole value function" (in the driving-home table below, each row has its own spread — that is exactly this point). And for a **fixed state $s$**, two different variances are hiding — one is the *root cause*, one is the *consequence*:

- **Variance of the learning target** $\mathrm{Var}(G_t)$: MC's update target is the return $G_t$, a random variable — for the same state, the $G_t$ sampled on different episodes (different days) varies a lot. It measures "how much the target you update toward fluctuates." This is the *root cause*.
- **Variance of the estimator** $\mathrm{Var}(\hat V(s))$: a trained $\hat V$ does output a fixed value for a fixed $s$ — but only *given one particular batch of training data*. Treat the data as random, and $\hat V(s)=\hat V(s;\text{random episodes})$ is again a random variable; retrain on a different batch of episodes and you learn a different $\hat V(s)$. Its variance is the spread **across many independent runs**. This is the *consequence*.

So my confusions resolve as: (1) one episode of course cannot give a variance — you have to **imagine driving the route many days**, collect the $G_t$ for the same state across all days, and see how spread out they are; (2) a trained $\hat V$ "has variance" not because its output for $s$ fluctuates, but because **the data fed to it is random** — different data, different $\hat V$; (3) throughout, variance is viewed **per state**, and different states' spreads can differ wildly (large for early states, small near the end).

For a random variable $X$, variance is formally defined as:

$$\mathrm{Var}(X) = \mathbb{E}\big[(X-\mathbb{E}[X])^2\big]$$

There is **no true value in this formula** — it only measures the spread of $X$ around **its own mean $\mathbb{E}[X]$**. "How far from the true value" is the **error (MSE)**, and it is the error that splits into bias and variance:

$$\underbrace{\mathbb{E}[(\hat V - v_\pi)^2]}_{\text{error (MSE)}} = \underbrace{(\mathbb{E}[\hat V]-v_\pi)^2}_{\text{bias}^2\;:\;\text{accuracy}} + \underbrace{\mathrm{Var}(\hat V)}_{\text{variance}\;:\;\text{stability}}$$

In one line: **variance = stability (how much it fluctuates around its own mean, independent of the true value); bias = accuracy (how far the mean is from the true value)**. Dartboard analogy: bias is how far the cluster's center is from the bullseye; variance is how scattered the darts are from each other.

### Variance: the Driving Home example

Start with the root cause — **the variance of the learning target**. Suppose we drive the route for three days (fast / medium / slow). For each state, list: the actual remaining time (= MC target $G_t$, the fast/medium/slow columns), their mean, the formula for the TD target $R_{t+1}+V(S_{t+1})$ (γ=1, with next-state estimates car 36, exit 20, 2ndary 12, home st 3, home 0), and each target's spread (a proxy for variance):

| State (early → late) | fast | med | slow | mean (≈ true value $V$) | TD target $=R_{t+1}+V(S_{t+1})$ | MC spread | TD spread |
|---|---|---|---|---|---|---|---|
| leaving office | 30 | 40 | 50 | 40 | $2/4/6 + 36$ | **±10** | **±2** |
| reach car | 28 | 36 | 44 | 36 | $14/16/18 + 20$ | ±8 | ±2 |
| exiting highway | 14 | 20 | 26 | 20 | $6/8/10 + 12$ | ±6 | ±2 |
| 2ndary road | 8 | 12 | 16 | 12 | $6/9/12 + 3$ | ±4 | ±3 |
| home street | 2 | 3 | 4 | 3 | $2/3/4 + 0$ | ±1 | ±1 |
| arrive home | 0 | 0 | 0 | 0 | $0 + 0$ | **0** | **0** |

Three things at a glance:

- **Both targets' means hit the true value**: both the MC target (fast/med/slow columns) and the TD target average to the true value $V$ per state — **both are unbiased**. (The TD target's per-day values are just "this segment's time + next-state estimate" from the formula, e.g. leaving office's $2/4/6+36 = 38/40/42$, again averaging to 40.)
- **MC spread is per-state and explodes toward the start (±1 → ±10)**: the $G_t$ from an early state accumulates the randomness of **every downstream segment**, and the earlier you are the more piles up; home street has only ~3 minutes left with little to vary (±1), and arrive home is already home (0).
- **TD spread barely changes with position (≈±2)**: the TD target only exposes the randomness of **this one segment**; the long future is taken over by the fixed estimate $V(S_{t+1})$. The two only coincide near the end (both ±1 — with one segment left, "actually driving it" equals "the accumulated remainder").

In one line: **the MC target exposes the randomness of the whole trajectory; the TD target exposes only one segment** — that is the mechanism by which bootstrapping suppresses variance. The price: if $V(S_{t+1})$ is inaccurate, the TD target is systematically off — the source of TD's bias.

## From Two Extremes to One Continuous Knob: TD(λ)

Driving Home only contrasts two extremes: MC waits all the way to the end, TD trusts just one step. But between them it is actually **continuous** — for how many steps does a single update use the real return before handing off to an estimate? That "search depth" is a knob you can turn.

- Use $n$ steps of real return, then bootstrap → **$n$-step TD**: $n=1$ is TD, $n\to\infty$ is MC.
- To avoid picking a specific $n$, mix all $n$ with geometric weights $(1-\lambda)\lambda^{n-1}$ into the **λ-return**, i.e. **TD(λ)**:

$$G_t^\lambda=(1-\lambda)\sum_{n\ge1}\lambda^{n-1}G_t^{(n)}$$

$\lambda=0$ degenerates to TD (the bias end), $\lambda=1$ degenerates to MC (the variance end), and $\lambda\in[0,1]$ interpolates continuously. So **λ is the "how much real future to fold in" knob**: larger is more MC-like (higher variance, but information propagates back along the state chain faster), smaller is more TD-like (lower variance, but slower propagation and more bias). The experiment below turns this knob from 0 to 1 while watching variance, bias, and convergence speed.

## Hands-On: Bias, Variance, and the λ Sweet Spot on a 19-State Random Walk

The above was rough arithmetic on a rather simple example. Let's switch to a classic **Markov Reward Process (MRP)** with a **known true value and analytically computable error**, and actually run it — the **19-state random walk** (the same example as Sutton & Barto Fig. 12.6): states $1\dots19$, terminating at both ends, starting in the center at $10$; step left/right with equal probability, +1 on entering the right end, −1 on the left, 0 otherwise; the true value is linear, $v(i)=(i-10)/10$ (i.e. $-0.9\dots+0.9$). (Note it is an MRP, not an MDP: there are **no actions/decisions** here; under a fixed policy there are only "state → reward" random transitions — pure policy evaluation. You need selectable actions to call it an MDP.)

![Structure of the 19-state random walk: states colored by true value, terminating ∓1 at both ends, starting in the center](https://cdn.jsdelivr.net/gh/pochenai/AI@main/rl/imgs/random-walk-diagram.svg)

The figure above is the structure of this MRP: 19 non-terminal states in a chain, terminal states at both ends (+1 on the right, −1 on the left), start dead center at $10$, each step going left/right with equal probability. States are colored by true value $v(i)$ (blue negative, red positive), symmetric from $-0.9$ to $+0.9$ — because the farther right you are, the more likely you hit the $+1$ end first.

The advantage of this example is that **the true value is analytic and error is exactly computable**, so we can ground the two variances from the Motivation: first the *root cause* (the distribution of the learning target), then the *consequence* (the bias and variance of the value-function estimator). Below we fix our gaze on one state: $s_0=15$, true value $+0.5$.

### Splitting Bias and Variance Apart — the Distribution of the Learning Target

Start with the *root cause*. For each λ, collect the **mean (→ bias)** and **standard deviation (→ variance)** of "the learning target $G_t^\lambda$ starting from $s_0$", drawn as error bars (point = mean, bar length = std):

![Mean (bias) and std (variance) of the learning target as λ varies](https://cdn.jsdelivr.net/gh/pochenai/AI@main/rl/imgs/bias-variance-targets.svg)

This figure deliberately uses two panels to separate "pure variance" from "the full trade-off":

- **(a) $V=$ true value (the estimate used for bootstrapping is accurate):** every λ's mean **sits right on the true-value line $+0.5$ (unbiased)**, but the error bars **grow longer as $\lambda\to1$** — the std climbs from $0.10$ at $\lambda{=}0$ to $0.87$ at $\lambda{=}1$. **This panel isolates "pure variance": bias is always 0, variance rises monotonically with λ**, because larger λ folds in more real future and accumulates more randomness.
- **(b) $V=0$ (the bootstrap estimate is untrained, inaccurate):** at $\lambda=0$ the target is dragged by bootstrapping to $\approx0$ (**huge bias**, the true value is $0.5$); as $\lambda\to1$ the mean **climbs back to the true-value line** (MC is unbiased regardless of whether $V$ is accurate), while the error bars lengthen in step. **This panel is the full bias-variance trade-off: larger λ, smaller bias, larger variance**.

In one line: variance rising monotonically with λ is its "nature" (panel a), while bias depends on whether the estimate that bootstrapping relies on is accurate (panel b) — that is why λ is a bias-variance knob.

### Pushing Bias and Variance to the "Estimator" Side — the Trained $\hat V(s_0)$

The previous figure dissected the distribution of the "update target" (the root cause); but what we ultimately care about is **how good the learned $\hat V$ is** (the consequence). So this time we actually train: **each run goes to convergence (300 episodes, $\alpha=0.05$), with 50 independent batches of data each trained once**, watching only $\hat V(s_0)$.

> **The most important point (a trap to avoid):** the variance here is the spread of the final $\hat V(s_0)$ **across 50 independent runs**, $\mathrm{Var}(\hat V(s_0))$ — **not** the within-run history of $\hat V(s_0)$ fluctuating over episodes. The latter mixes in the "not-yet-converged" transient and is not the estimator variance in the decomposition. Bias, likewise, is **the mean of the final values across runs** minus the true value. Also, the episode count must be large enough: otherwise TD has not learned the true value yet, and that "bias" is just an under-training artifact, not the algorithm's own bias.

![Bias and estimator variance of the trained V̂(s0) as λ varies](https://cdn.jsdelivr.net/gh/pochenai/AI@main/rl/imgs/estimator-bias-variance-vs0.svg)

- **Panel (a) — the convergence process (mean line ± 1 std band):** **large λ converges fast but the band is wide and fluctuates a lot** ($\lambda=1$ reaches the true-value line within a few dozen episodes, but has the widest band); **small λ converges slowly but the band is narrow** ($\lambda=0$ takes three hundred episodes to gradually climb to $0.5$, but barely fluctuates). This is exactly the two faces of the λ knob: "fast propagation ↔ high variance."
- **Panel (b) — the final $\hat V(s_0)$ across runs after convergence (point = mean, bar length = std):** **every λ's point lands near the true value $0.5$ (essentially unbiased after convergence)**, and the difference is **pure variance** — the error bars **lengthen monotonically with λ** (std from $0.04$ at $\lambda{=}0$ to $0.49$ at $\lambda{=}1$). This is the estimator-side confirmation of panel (a) above: once training removes the bias, all that is left is the "λ↑ → variance↑" pure-variance axis.

**The sweet spot can be read straight off panel (a):** large λ shoots up fast at first (small early error) but has large steady-state variance; small λ is slow early but the most stable in steady state. So under **any finite training budget**, total error = bias² + variance: each extreme has a weakness (small λ has large early bias, large λ has large variance throughout), and **only a middle λ gets both "fast propagation" and "controlled variance," minimizing total error** — that is the sweet spot (in this random walk, early-learning setup, comparing each λ at its own best step size α, the measured optimum is about **λ≈0.6**; the exact location shifts with training budget and α). It is not a mystical hyperparameter, but the knob's optimal compromise point on the "bias ↔ variance" axis.

## Summary

**Variance** measures how much the estimator $\hat V(s)$ "fluctuates around its own mean" across many independent training runs, independent of how far it is from the true value; it has two layers — a root cause (the learning target) and a consequence (the estimator). And **λ (search depth)** is the continuous knob that adjusts bias and variance together: the two trade off against each other, and the optimum almost always lands somewhere between the two extremes (about λ≈0.6 in this setup).

A couple of caveats:

- **The bias-for-variance trade lives on the Markov property.** TD uses $V(S_{t+1})$ to take over the future, on the premise that "the return after reaching $S_{t+1}$ depends only on $S_{t+1}$." The closer the state is to Markov, the more worthwhile this "trade bias for variance" is; once it is a POMDP (state drops history), bootstrapping introduces **systematic bias**, and you should retreat toward the MC end.
- **Under function approximation, which of MC and TD is more accurate can flip.** In the tabular case every λ is asymptotically unbiased; but with function approximation, MC converges to "the closest projection to the true value" (minimum error), while TD converges to the TD fixed point, which can be amplified by up to a factor of $\tfrac{1}{1-\gamma}$. Here **MC is actually more accurate**, and TD merely trades bias for speed and low variance.

## Reproduce

All figures in this post (the random walk structure, the target distribution, the estimator bias/variance) are generated by a single script that depends only on numpy and matplotlib:

```bash
git clone https://github.com/pochenai/AI.git
cd AI/rl/py_codes
python3 td_lambda_study.py   # three SVGs are written to ../imgs/
```
