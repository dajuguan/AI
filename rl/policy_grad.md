# Policy Gradient

## 如何度量策略好坏:三种目标函数 $J(\theta)$

Policy gradient 的目标是"找最好的 $\theta$",但先得定义**什么叫策略好** —— 即给出一个标量目标 $J(\theta)$。不同环境类型对应三种常见定义。

### 1. Start value(起始值)—— 适用于 episodic 环境

$$J_1(\theta) = V^{\pi_\theta}(s_1) = \mathbb{E}_{\pi_\theta}[v_1]$$

从固定起始状态 $s_1$ 出发,策略能拿到的**期望总回报**。适合每个 episode 都从同一个(或固定分布的)起点开始、且有终止的任务(如走迷宫、下棋)。含义最直白:**"从头玩一局,平均能得多少分"**。

### 2. Average value(平均值)—— 适用于 continuing 环境

$$J_{avV}(\theta) = \sum_s d^{\pi_\theta}(s)\,V^{\pi_\theta}(s)$$

没有自然起点/终点的持续任务里,没有唯一的 $s_1$ 可用。于是对**所有状态的值 $V^{\pi_\theta}(s)$ 做加权平均**,权重是 $d^{\pi_\theta}(s)$ —— 即策略 $\pi_\theta$ 诱导的马尔可夫链的**平稳分布**(长期来看待在各状态的时间占比)。含义:**"按实际停留频率加权,平均每个状态有多好"**。

### 3. Average reward per time-step(每步平均奖励)—— 适用于 continuing 环境

$$J_{avR}(\theta) = \sum_s d^{\pi_\theta}(s)\sum_a \pi_\theta(s,a)\,\mathcal{R}_s^a$$

同样按平稳分布 $d^{\pi_\theta}(s)$ 加权,但看的是**每一步的即时奖励期望**(先按 $\pi_\theta$ 选动作,再取奖励均值)。含义:**"长期运行下,平均每个时间步能拿多少即时奖励"**。

### avV vs avR:两种 continuing 目标分别用在哪

两者都用平稳分布 $d^{\pi_\theta}(s)$ 加权,区别在于**要不要折扣、关心「累积价值」还是「稳态速率」**。

**$J_{avR}$(每步平均奖励)—— 关心速率 / 吞吐**,通常配无折扣 $\gamma=1$,衡量"长期运行下每步的即时奖励速率"。适合永远跑、没有终点、关心单位时间产出且未来与现在同等重要的任务:

- 网络路由 / 通信:最大化长期吞吐量
- 排队 / 资源调度:稳态下每步平均服务量、等待代价
- 生产线 / 机器人持续作业:每小时产量、每步能耗-收益
- 做市 / 高频策略:长期每步平均收益率

**$J_{avV}$(平均值)—— 关心折扣累积价值**,$V^{\pi_\theta}(s)$ 是带折扣 $\gamma<1$ 的累积回报。适合任务持续但仍想折扣的场景:

- 近期收益优先、需体现时间价值的持续控制或金融场景
- 想复用已有折扣值函数框架(TD、Q 等多基于 $\gamma<1$)
- 折扣起正则化作用,让无穷时域的值有界、估计更稳

| | $J_{avR}$ | $J_{avV}$ |
|---|---|---|
| 折扣 | 一般 $\gamma=1$ | $\gamma<1$ |
| 衡量对象 | 即时奖励的稳态速率 | 折扣累积回报的加权平均 |
| 你在问 | "平均每步赚多少" | "折扣后长期总共值多少" |
| 典型场景 | 吞吐/调度/持续作业的速率优化 | 需时间折扣的持续控制 |

在稳态下两者密切相关(avV 大致是 avR 除以 $1-\gamma$ 量级),选哪个主要看**任务该不该折扣、你关心累积量还是速率**。

### 关键概念与联系

- **平稳分布 $d^{\pi_\theta}(s)$**:固定策略 $\pi_\theta$ 后,状态转移构成一条马尔可夫链;$d^{\pi_\theta}$ 是它的平稳分布,表示"长期在状态 $s$ 上花的时间比例"。continuing 目标全靠它把"无穷长的过程"压成一个有限加权和。注意它**依赖 $\theta$** —— 改策略会改变你停留在哪些状态,这也是策略梯度定理里需要小心处理的一项。
- **avV vs avR**:一个加权"长期值 $V$",一个加权"即时奖励 $\mathcal{R}$"。在持续任务里两者密切相关(平均值可由每步平均奖励在平稳态下推出),常被视作等价的连续任务目标。
- **最重要的一点**:无论选哪个 $J(\theta)$,**策略梯度的形式都一样**:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(s,a)\,Q^{\pi_\theta}(s,a)\big]$$

这就是策略梯度定理。所以三种指标只是"衡量好坏的尺子"不同(episodic 用起点回报、continuing 用平稳态平均),但落到怎么求梯度、怎么更新参数,是统一的 —— 具体形式见下文 one-step MDP 的推导。

## One-Step MDP:为什么梯度要写成期望形式

先看最简单的 one-step MDP:从 $s\sim d(s)$ 出发,走一步、拿到奖励 $r=\mathcal{R}_{s,a}$ 后终止。目标是期望奖励:

$$J(\theta) = \mathbb{E}_{\pi_\theta}[r] = \sum_{s\in\mathcal{S}} d(s)\sum_{a\in\mathcal{A}} \pi_\theta(s,a)\,\mathcal{R}_{s,a}$$

### 求导

$d(s)$(起始分布)与 $\mathcal{R}_{s,a}$(环境奖励)都不依赖 $\theta$,梯度只落在 $\pi_\theta$ 上:

$$\nabla_\theta J(\theta) = \sum_{s} d(s)\sum_{a} \nabla_\theta\pi_\theta(s,a)\,\mathcal{R}_{s,a}$$

此时若直接留着 $\nabla_\theta\pi_\theta(s,a)$,它**不是任何概率分布下的期望**($\nabla\pi$ 对 $a$ 求和为 0,不能当采样权重)。用 **likelihood-ratio(log)技巧**塞回一个 $\pi_\theta$ 因子:

$$\nabla_\theta \pi_\theta(s,a) = \pi_\theta(s,a)\,\nabla_\theta\log\pi_\theta(s,a)$$

代回:

$$\nabla_\theta J(\theta) = \sum_{s} d(s)\sum_{a} \pi_\theta(s,a)\,\nabla_\theta\log\pi_\theta(s,a)\,\mathcal{R}_{s,a} = \mathbb{E}_{\pi_\theta}\!\big[\nabla_\theta\log\pi_\theta(s,a)\,r\big]$$

### 为什么要转成期望

权重 $\sum_s d(s)\sum_a \pi_\theta(s,a)[\cdots]$ 恰好是"按 $d(s)$ 抽状态、按 $\pi_\theta$ 抽动作"的联合分布,所以整体就是该分布下的期望。log 技巧的意义就是**把 $\nabla\pi$ 变成 $\pi\times\nabla\log\pi$,让 $\pi$ 当采样权重,从而把梯度装进期望**。

一旦是期望,就不必显式知道下列任何一项:

| 求和形式需要 | 期望形式只需要 |
|---|---|
| 已知 $d(s)$ 才能加权 | 跑策略,状态自然按 $d(s)$ 出现 |
| 遍历所有 $s,a$ | 只用采样到的 $(s,a)$ |
| 已知奖励模型 $\mathcal{R}_{s,a}$ | 直接用观测到的 $r$ |

于是估计梯度就变成"采样—平均"(无偏估计):

$$\nabla_\theta J(\theta) \approx \frac{1}{N}\sum_{i=1}^N \nabla_\theta\log\pi_\theta(s_i,a_i)\,r_i$$

只要**运行当前策略**收集 $(s_i,a_i,r_i)$ 即可 —— 这正是 model-free 策略梯度能落地的关键。

## Softmax 策略的 score function 推导

Policy gradient 用 softmax 策略作为常用示例:动作权重是特征的线性组合 $\phi(s,a)^\top\theta$,动作概率正比于其指数化权重。

$$\pi_\theta(s,a) = \frac{e^{\phi(s,a)^\top\theta}}{\sum_b e^{\phi(s,b)^\top\theta}}$$

### 缺失拼图:$\theta$ 是怎么"牵动"概率的

一个常见卡点:$\theta$ 和"每个动作的概率"看起来脱节。补上中间的映射就通了 —— **$\theta$ 不直接是概率,而是先算出每个动作的"吸引力分数",再由 softmax 把分数映射成概率**:

$$\theta \xrightarrow{\ \phi(s,a)^\top\theta\ } \underbrace{\text{吸引力分数}}_{\text{每个动作一个实数}} \xrightarrow{\ \text{softmax}\ } \pi_\theta(s,a)$$

- **第一步(打分)**:动作的属性画像 $\phi(s,a)$ 固定,$\theta$ 是"你对各属性的偏好";两者内积 $\phi(s,a)^\top\theta$ 给出该动作的吸引力分数。
- **第二步(映射为概率)**:softmax 把分数指数化再归一化,分数越高频率越高、但都有份。这一步就是之前容易漏掉的 missing piece。

于是"调 $\theta$"→"改吸引力分数"→"softmax 自动重排所有动作的概率";归一化($\sum_a\pi=1$)保证此消彼长。也正因为有归一化这一步,下面求导才会冒出 $-\mathbb{E}_\pi[\phi]$ 那一项。

> 夹菜版:$\theta$ = 你对辣/甜/肉的偏好,$\phi$ = 每道菜的属性,内积得每道菜的吸引力,softmax 把吸引力换算成夹取频率。你从不手动设"这道夹几次",全由 softmax 换算。

### 推导

取 log,拆成分子与分母(配分函数)两部分:

$$\log\pi_\theta(s,a) = \underbrace{\phi(s,a)^\top\theta}_{\text{分子}} - \underbrace{\log\sum_b e^{\phi(s,b)^\top\theta}}_{\text{分母 / 配分函数}}$$

对 $\theta$ 求梯度。

**第一项**(线性,直接):

$$\nabla_\theta\,\phi(s,a)^\top\theta = \phi(s,a)$$

**第二项**(log-sum-exp,链式法则):

$$\nabla_\theta \log\sum_b e^{\phi(s,b)^\top\theta}
= \frac{\sum_b e^{\phi(s,b)^\top\theta}\,\phi(s,b)}{\sum_b e^{\phi(s,b)^\top\theta}}
= \sum_b \pi_\theta(s,b)\,\phi(s,b)
= \mathbb{E}_{\pi_\theta}[\phi(s,\cdot)]$$

关键在中间那步:分子分母一约,权重恰好还原成 $\pi_\theta(s,b)$,于是加权平均就是**当前策略下特征的期望**。

合并两项即得 score function:

$$\boxed{\;\nabla_\theta\log\pi_\theta(s,a) = \phi(s,a) - \mathbb{E}_{\pi_\theta}[\phi(s,\cdot)]\;}$$

### 为什么有减项 —— 直觉

因为概率是**相对**的,不是绝对的。减项来自 softmax 分母(归一化)的求导,等于当前策略下的平均特征,充当 baseline:

- $\phi(s,a)$:把**实际选中动作** $a$ 的特征方向往上顶;
- $-\mathbb{E}_{\pi_\theta}[\phi(s,\cdot)]$:减去所有动作特征的平均。

所以 score function 衡量的是"**这个动作的特征比平均动作高出多少**":高于均值 → 梯度为正,提高其概率;低于均值 → 降低。这也保证概率和恒为 1 —— 推高一个动作必然压低其它,减项正是归一化约束在梯度里的体现。

一个漂亮的性质:

$$\mathbb{E}_{\pi_\theta}[\nabla_\theta\log\pi_\theta] = \mathbb{E}[\phi] - \mathbb{E}[\phi] = 0$$

score function 期望为零 —— 这是所有合法 score function 的通性,也是策略梯度中引入 baseline 减方差的基础。

## score 项如何"改变动作的出现概率"

策略梯度里 $\nabla_\theta\log\pi_\theta(s,a)$ 指向"让动作 $a$ 更可能被选"的方向,再用 $Q$ 当权重。但这个"往梯度方向推、按 $Q$ 加权"是**在采样里做的**,而采样本身就带着"出现概率"这个偏差 —— $\nabla\log\pi$ 里的 $\frac{1}{\pi}$ 恰好把它抵消,于是净效果直接变成"对概率 $\pi$ 本身求梯度"。

### 两个"概率"要分清

1. **策略概率** $\pi_\theta(s,a)$:策略给动作 $a$ 分配的概率。
2. **采样频率**:真跑策略时 $(s,a)$ 实际出现的频率 —— 它**正好等于** $\pi_\theta(s,a)$。

矛盾点:概率大的动作被采样到的次数也多。若只是"每见到一次就往它方向推一下",高概率动作会被无脑越推越高,显然不对。

### $1/\pi$ 抵消采样偏差

由恒等式 $\pi_\theta(s,a)\,\nabla_\theta\log\pi_\theta(s,a) = \nabla_\theta\pi_\theta(s,a)$,即 $\nabla\log\pi=\frac{\nabla\pi}{\pi}$,自带一个 $\frac{1}{\pi}$。把期望展开:

$$\nabla_\theta J = \mathbb{E}_{a\sim\pi_\theta}\big[\nabla_\theta\log\pi_\theta(s,a)\,Q\big] = \sum_a \underbrace{\pi_\theta(s,a)}_{\text{出现频率}}\cdot\underbrace{\nabla_\theta\log\pi_\theta(s,a)}_{(1/\pi)\nabla\pi}\cdot Q = \sum_a \nabla_\theta\pi_\theta(s,a)\,Q(s,a)$$

最后一步:$\pi$(采样频率)与 $\nabla\log\pi$ 里的 $\frac{1}{\pi}$ 约掉,只剩 $\sum_a \nabla\pi(s,a)\,Q(s,a)$ —— 一句纯粹关于**概率如何变化**的话:"按 $Q$ 加权,直接调整每个动作的概率 $\pi(s,a)$"。

```
采样中"见得多的动作被推得多"(偏差 = π)
        × ∇log π 自带 1/π(反偏差)
        ↓ 相乘 = ∇π
净效果 = "对概率求梯度"
```

### 具体例子:为什么必须有 $1/\pi$

状态 $s$,两个动作,softmax 下 $\pi(a_1)=0.9,\ \pi(a_2)=0.1$,且两者一样好 $Q(a_1)=Q(a_2)=1$。直觉上策略不该变。

- **没有 $1/\pi$**(错):$a_1$ 被采到次数是 $a_2$ 的 9 倍,每次都往采到的动作推,$a_1$ 概率被越推越高 —— 错。
- **有 $1/\pi$**(对):
$$\sum_a \nabla\pi(s,a)\,Q = Q\sum_a\nabla\pi(s,a) = Q\,\nabla_\theta\Big(\sum_a\pi(s,a)\Big) = Q\,\nabla_\theta(1) = 0$$
净更新为 0,策略不动。✅ 高概率动作虽被采样 9 倍多,但每次更新被 $\frac{1}{\pi}$ 缩小,刚好抵消。

### 与归一化约束的联系

softmax 下 $\nabla_\theta\log\pi = \phi(s,a) - \mathbb{E}_{\pi}[\phi(s,\cdot)]$,减项(baseline)保证:**推高一个动作的概率,其它动作必然被压低**,因为 $\sum_a\pi=1$ 是硬约束。上例中 $\sum_a\nabla\pi=0$ 正是这个约束的体现 —— 概率是此消彼长地被重新分配的。

> **一句话**:"往 $a$ 方向推、按 $Q$ 加权"之所以能正确改变**出现概率**,是因为 $\nabla\log\pi$ 里的 $\frac{1}{\pi}$ 抵消了"概率大的动作被采样多"的偏差;$\pi\cdot\nabla\log\pi=\nabla\pi$ 让更新净化成"直接对概率 $\pi$ 求梯度",再由归一化把提升某动作转成对其它动作的相对压低。

## 更新 $\theta$ 与"按 $Q$ 加权抬概率"是同一件事

一个常见困惑:我们明明在更新参数 $\theta$、目的是让 $J$ 变大,为什么最后表现成"按 $Q$ 加权提高每个动作的概率"?——因为这两者是**同一个向量的两种读法**,不是两件事。

### 你不能直接调概率,只能拧 $\theta$

概率 $\pi_\theta(s,a)$ 是 $\theta$ 的函数(如 softmax),你能动的只有 $\theta$。链条是:

$$\theta \;\longrightarrow\; \pi_\theta(s,a) \;\longrightarrow\; J(\theta)$$

### 链式法则:把"调 $\theta$"翻译成"调概率"

$$\nabla_\theta J = \sum_a \underbrace{\frac{\partial J}{\partial \pi_\theta(s,a)}}_{\text{概率变}\to J\text{变多少}}\cdot\underbrace{\nabla_\theta\,\pi_\theta(s,a)}_{\theta\text{变}\to\text{概率变多少}}$$

对 one-step 的 $J=\sum_a \pi_\theta(s,a)Q(s,a)$,第一截恰好 $\dfrac{\partial J}{\partial \pi_\theta(s,a)}=Q(s,a)$,于是:

$$\nabla_\theta J = \sum_a Q(s,a)\,\nabla_\theta\pi_\theta(s,a)$$

这个式子同时是两句话:左边"$J$ 对 $\theta$ 的梯度"(要沿它更新 $\theta$),右边"每个动作概率的变化 $\nabla\pi$,按其对 $J$ 的贡献 $Q$ 加权"。所以**更新 $\theta$ 与按 $Q$ 加权调概率是同一个向量**;$Q$ 当权重,是因为它就是"该动作概率每涨一点、$J$ 能涨多少"的灵敏度。

### 为什么这么调就能让 $J$ 变大

**数学定义**:梯度上升 $\theta\leftarrow\theta+\alpha\nabla_\theta J$,小步长下 $J(\theta+\alpha\nabla J)\approx J(\theta)+\alpha\|\nabla J\|^2\ge J(\theta)$,$J$ 必然(局部)增大。

**概率直觉**:在一个状态里 $J$ 就是 $Q$ 的概率加权平均

$$J = \sum_a \pi_\theta(s,a)Q(s,a) = \mathbb{E}_{a\sim\pi}[Q]$$

想让加权平均变大、而你能动的是权重,那自然是把概率从低 $Q$ 动作挪向高 $Q$ 动作。梯度做的正是这件事,$\theta$ 只是挪动权重的手柄。

> **夹菜比喻**:$J$ 是一桌菜的"平均好吃程度",概率是夹每道菜的频率,$Q$ 是每道菜多好吃。你不能直接规定"多夹这道",只能调口味参数 $\theta$;但只要朝"让平均更好吃"调 $\theta$,结果一定表现为多夹好吃的、少夹难吃的。归一化($\sum\pi=1$,总共只能夹这么多次)保证多夹一道就得少夹另一道。

**一句话**:更新 $\theta$ 是手段,让 $J$ 增大是目的;由于 $J=\mathbb{E}_\pi[Q]$ 而概率由 $\theta$ 决定,链式法则 $\nabla_\theta J=\sum_a Q\,\nabla_\theta\pi$ 自动把"沿 $J$ 上升调 $\theta$"翻译成"把概率往高 $Q$ 动作挪"——是同一动作的因果两端。

### 两个易绊的点(踩坑提醒)

**① $\pi_\theta(s,a)$ 的自变量是 $(s,a)$,$\theta$ 只是参数(旋钮)。**
容易把"更新 $\theta$"误解成"更新 $\pi$ 里的某个参数",从而觉得它和"$\pi(s,a)$ 的值怎么变"脱了节。正确的图景是:$\theta$ 是手柄,$\pi(s,a)$ 是被它牵动的输出——

$$\theta \xrightarrow{\text{手柄}} \pi_\theta(s,a) \xrightarrow{\text{链式}} J$$

一旦这么看,整条因果链就顺了:更新 $\theta$ → 牵动 $\pi(s,a)$ 按 $Q$ 加权移动 → 经链式法则与 $J$ 增大方向一致 → $J$ 增大。

**② 不是每个 $\pi(s,a)$ 都增大,而是概率此消彼长地重新分配。**
因为 $\sum_a\pi(s,a)=1$ 是硬约束,抬高好动作必然压低差动作:

- $Q$ 高于当前均值的动作 → 概率上升
- $Q$ 低于当前均值的动作 → 概率下降
- 净效果:概率质量整体往高 $Q$ 挪 → $J=\mathbb{E}_\pi[Q]$ 变大

(这正是 softmax score 里 $-\mathbb{E}_\pi[\phi]$ baseline 项的作用:把"绝对抬高"变成"相对于平均的增减"。)

## State aliasing:为什么要直接学策略

State aliasing(感知混叠)指**两个不同的真实状态,在 agent 的观测 $\phi(s)$ 下完全相同**,无法区分。这是部分可观测(POMDP)下的典型情形,也是 policy gradient 相对 value-based greedy 策略的一个结构性优势论点。

### 经典走廊例子(David Silver)

单行走廊,agent 的观测是**四个方向(N/E/S/W)有没有墙**,不含绝对位置、不含历史记忆:

```
┌───┬───┬───┬───┬───┐
│   │ ▓ │   │ ▓ │   │
└───┴───┴───┴───┴───┘
  1    2   3   4   5
```

- 上下(N/S)每格都是墙,提供不了信息;
- 左右(E/W)只在两端有墙,中间格子左右都空。

于是第 2 格与第 4 格观测都是 `[N=墙, S=墙, E=空, W=空]`,**完全相同 → aliased**。

### 后果:随机策略胜出

假设最优是第 2 格向右、第 4 格向左,但两格观测相同:

- **确定性策略**:对同一观测只能输出一个动作,总有一个灰格会卡死(来回撞墙,永远够不到目标)。
- **随机策略**(policy gradient 可学到):在该观测下各 50% 向左/向右,无论身处哪个灰格都有机会逃出。

> 在 MDP 中一定存在最优确定性策略;但在**有 state aliasing 的部分可观测情形下,最优策略可能是随机的**。policy gradient 能自然表示并收敛到随机策略,而 value-based greedy 结构上做不到。

### 关于 lookahead

Aliasing 是**相对于 agent 决策所用信息**定义的:

- 给 observation 加坐标 / 记忆 / 更好传感器(feature augmentation)→ 两格输入不同 → 直接消灭 aliasing,但那已是"换了个更丰富的表示",跳出了该 case;
- 纯 model-based 前瞻(输入信息不变,只模拟"走一步会看到啥")→ 因为例子布局**关于中心格对称**,模拟出的邻居观测也是 aliased 的,救不了。

经典例子刻意**锁死观测(只有局部墙形状)+ 对称布局**,就是为了堵死"加点信息就能确定性最优"这条路,从而干净地凸显:随机策略 > 确定性策略。
