"""
Distributed Data Parallelism: DDP -> ZeRO-1 -> ZeRO-2 -> ZeRO-3 / FSDP

一个很有用的主线是：
    DDP 先把所有状态都复制一遍，追求实现简单。
    ZeRO 再一步步把 optimizer / grad / param 分片，追求省显存。
    FSDP 可以把它看成工程化很成熟的 ZeRO stage 3。


1. DDP (Distributed Data Parallel)

每张卡都有：
    - 完整 param
    - 完整 grad
    - 完整 optimizer state

训练流程：
    每个 GPU:
        forward
        backward
        all-reduce gradients   -> 每张卡最后都有完整 grad
        optimizer step         -> 每张卡都用完整 (m, v) 更新完整参数

直觉：
    - 实现最直接。
    - 但显存最贵，因为模型状态被完整复制了 N 份。


2. ZeRO stage 1: shared optimizer

每张卡都有：
    - 完整 param
    - 完整 grad
每张卡只有：
    - optimizer state 的 1/N

训练流程：
    每个 GPU:
        forward
        backward
        all-reduce gradients   -> 每张卡都有完整 grad

    每个 GPU:
        只更新自己负责的那部分参数（只需要自己那份 m, v）

    然后：
        all-gather parameters  -> 把更新后的参数同步给所有 GPU

直觉：
    - 先把最占内存的 optimizer state 切开。
    - 参数和梯度仍然是全量复制。


3. ZeRO stage 2: shared gradients by layer / bucket

每张卡都有：
    - 完整 param
每张卡只有：
    - grad 的 1/N
    - optimizer state 的 1/N

核心思想：
    - DDP 的问题是：每张卡都保留完整 grad，太浪费显存。
    - stage 2 的做法是：某一层的梯度一算出来，就立刻做 reduce-scatter。
    - reduce-scatter 之后，每张卡只保留自己负责的那 1/N 梯度分片，
      不再保留该层的完整 grad。

训练流程：
    每个 GPU:
        forward   (因为 param 仍然是完整的，所以 forward 和 DDP 一样)

    backward 按 layer / bucket 逐步进行：
        对于当前层:
            1. 先算出本卡的 local grad
            2. 对该层 grad 做 reduce-scatter
               -> 聚合后，每张卡只拿到自己负责的 grad shard
            3. 立刻释放这一层的 full grad buffer

    每个 GPU:
        用自己的 grad shard + optimizer shard
        只更新自己负责的参数分片

    然后：
        all-gather updated parameter shards
        -> 恢复出下一轮 forward 需要的完整参数

为什么 stage 2 的通信量通常没有比 DDP 更大：
    - DDP 做的是 all-reduce(grad)。
    - 而 all-reduce 本质上可以分解成：
        reduce-scatter + all-gather
    - stage 2 把 DDP 的那次“大一统 all-reduce”拆开了：
        backward 时做 reduce-scatter(grad)
        update 之后做 all-gather(param)
    - 所以从总字节数的量级看，stage 2 和 DDP 是同一个数量级，
      不是突然多出一个数量级的通信。

stage 2 的收益：
    - 参数还是完整复制，所以 forward 很简单。
    - 但 grad + optimizer 都分片了，显存已经明显下降。


4. ZeRO stage 3 / FSDP: shard parameters too

每张卡只有：
    - param 的 1/N
    - grad 的 1/N
    - optimizer state 的 1/N

核心思想：
    - stage 2 里 param 仍然是全量复制，超大模型时还是放不下。
    - stage 3 / FSDP 进一步把 param 也切开。
    - 真正计算某个 module 之前，临时把这个 module 的 full param 聚起来；
      用完之后再把 full param 释放掉。
    - 也就是说：full weights 只在“当前正在算的 module”上短暂存在，
      而不是整模型一直常驻。

forward 流程（按 FSDP unit / module）：
    对于当前 module:
        1. all-gather parameter shards
           -> 临时拼出这个 module 的 full param
        2. 执行该 module 的 forward
        3. 如果采用 reshard-after-forward:
           立刻丢掉 full param，只保留本卡 shard

backward 流程（反向按 module 倒序）：
    对于当前 module:
        1. 如果 forward 后已经 reshard 了，
           需要再次 all-gather parameter shards
           -> 为了能正确计算该 module 的 backward
        2. 执行该 module 的 backward
        3. 对该 module 的 grad 做 reduce-scatter
           -> 每张卡只保留自己负责的 grad shard
        4. 释放 full param / full grad 的临时 buffer

optimizer step：
    每个 GPU:
        只对自己负责的 param shard
        用自己负责的 grad shard + optimizer shard 做更新

直觉：
    - DDP:      replicate everything
    - ZeRO-2:   replicate params, shard grads + optimizer
    - ZeRO-3:   shard everything, only materialize the current layer/module


5. 一组很好记的通信量数字（忽略 latency / bucket 细节）

记：
    P = 模型参数量（#params）

如果只做 slide-level 的粗略 bandwidth accounting，
一组很好记的数字是：
    - DDP ~= 2P communication
    - ZeRO stage 1 ~= 2P
    - ZeRO stage 2 ~= 2P
    - ZeRO stage 3 / FSDP ~= 3P

为什么是这几个数：
    - DDP:
        all-reduce(grad) ~= reduce-scatter + all-gather ~= 2P
    - ZeRO stage 1:
        只是把 optimizer state shard 掉，
        主通信仍然基本就是 DDP 那次梯度同步，所以 ~= 2P
    - ZeRO stage 2:
        backward 时 reduce-scatter(grad)
        + update 之后 all-gather(param)
        ~= 2P
    - ZeRO stage 3 / FSDP:
        在 stage 2 的基础上，
        还要为真正计算某个 module 临时 all-gather(param)
        所以粗记成 ~= 3P

所以：
    - stage 1 基本可以看成“白送”的显存优化
    - stage 2 通常也接近“几乎白送”（忽略额外 overhead）
    - stage 3 / FSDP 相对 DDP 的通信量大约是:
        3P / 2P = 1.5x

注意：
    - 这组 2P / 3P 是一个很好用的记忆公式，
      不是逐条 collective 的精确 opcode 计数。
    - 更细的 FSDP 实现里，你会看到 reshard / prefetch / overlap /
      bucket 等细节；但从总字节量的粗略量级看，
      把 stage 3 记成 `3P` 很有帮助。


6. 为什么 FSDP 的 overhead 实际上通常没有那么大

很多人第一次看 FSDP 会觉得：
    “每层都要 all-gather 一次，这不是会慢很多吗？”

这个担心不是完全错，但通常会高估它的真实 wall-clock 开销。

原因 1：从 slide-level 粗算看，它只是从 2P 变成 3P
    - 也就是相对 DDP 大约 1.5x 的通信量。
    - 这当然不是免费，但也远不是数量级爆炸。

原因 2：all-reduce 本来就等价于 reduce-scatter + all-gather
    - 所以 ZeRO/FSDP 并不是引入了一个完全陌生的新代价，
      而是在重组 DDP 原本就要付的 collective communication。
    - 很多教材里看起来“操作变多了”，
      但底层常常还是那些带宽最优的 collective 原语。

原因 3：更细的实现细节里，多出来的通信并不都暴露在关键路径上
    - 直白地画执行图时，FSDP 会出现 per-module 的 all-gather /
      reshard / reduce-scatter。
    - 但真实实现通常会做 prefetch 和 overlap，
      所以“看到更多通信原语”不等于 wall-clock 就线性变慢。

原因 4：只 materialize 当前 module，显存压力大幅下降
    - 这点非常关键。
    - 如果显存压力小很多，我们往往可以：
        用更大的模型
        用更大的 micro-batch
        减少某些更痛苦的 offload / activation recomputation
    - 这样带来的计算效率提升，常常会抵掉一部分 FSDP 的额外通信。
    - 换句话说，FSDP 虽然多了一些通信，但它换来了更高的“可训练规模”。

原因 5：真实实现不会傻傻地“一层一个超小消息”
    - 工程里通常会做 flat parameter / bucket 化。
    - 这样可以把很多小 tensor 合并成更大的通信块，
      避免被 message latency 主导。
    - 所以真实开销往往比“每层一次网络往返”这个直觉更小。

一个好用的心智模型：
    - DDP 的代价：把整模型复制到每张卡上，然后统一同步 grad。
    - FSDP 的代价：不再长期复制整模型，而是在需要某个 module 时临时把它聚起来。
    - 前者省通信设计复杂度，后者省显存。
    - FSDP 付出的不是“灾难性额外开销”，而是
      “为了极大显存节省而接受的一笔通常可控的通信常数项”。


7. 一组很好记的显存估算数字（8x A100 80G）

下面这组数字依赖一个具体假设：
    - 机器：8 x A100 80GB
    - 训练：pure BF16 training（带 Kahan summation）
    - 存储：除了 master weights 用 FP32 以外，其他尽量用 BF16

按这组假设，可以粗记成：
    - param: 2 bytes
    - grad: 2 bytes
    - master weights: 4 bytes
    - optimizer state: 4 bytes

所以 baseline 总共大约是：
    2 + 2 + 4 + 4 = 12 bytes / param

于是每种方案的 bytes / param 可以粗算为：

    Baseline / DDP:
        12 bytes / param
        -> 单卡最多容纳 80 / 12 = 6.67B params

    ZeRO stage 1:
        2(param) + 2(grad) + (4 master + 4 optimizer) / 8
        = 5 bytes / param
        -> 单卡最多容纳 80 / 5 = 16B params

    ZeRO stage 2:
        2(param) + (2 grad + 4 master + 4 optimizer) / 8
        = 2 + 10 / 8
        = 3.25 bytes / param
        -> 单卡最多容纳 80 / 3.25 = 24.62B params

    ZeRO stage 3 / FSDP:
        (2 param + 2 grad + 4 master + 4 optimizer) / 8
        = 12 / 8
        = 1.5 bytes / param
        -> 单卡最多容纳 80 / 1.5 = 53.33B params

把它们并排记，会很有感觉：
    - Baseline / DDP: 12 B/param  ->  6.67B params
    - ZeRO-1:          5 B/param  -> 16.00B params
    - ZeRO-2:       3.25 B/param  -> 24.62B params
    - ZeRO-3/FSDP:   1.5 B/param  -> 53.33B params

这组数字想说明的重点不是“小数点后两位有多神圣”，
而是：
    - stage 1 就已经能明显扩大可训练模型规模
    - stage 2 继续省下大块 grad 内存
    - stage 3 / FSDP 会带来最剧烈的容量提升


8. FSDP 什么时候 overhead 会真的变大

下面这些场景下，FSDP 的收益会变差，甚至可能比 DDP 慢不少：
    - 模型很小：计算量不足以隐藏通信
    - batch 太小：compute/communication ratio 太差
    - interconnect 很慢：比如网络带宽差、拓扑不理想
    - wrap 太碎：切成太多很小的 FSDP units，导致很多小消息
    - 开了 CPU offload：PCIe / host-device copy 代价会明显上来
    - overlap / prefetch 没配好：通信暴露在关键路径上

所以一句话总结：
    FSDP 不是免费的，但它的 overhead 往往只是“可控的常数倍”，
    而不是直觉上那种“每层 gather 一次所以一定巨慢”。
    对大模型训练来说，省下来的显存通常值回票价。
"""
