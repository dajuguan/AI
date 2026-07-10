'''扔鸡蛋问题，主要思考的是状态转移方程(action-state)的设计，以及怎么extract policy.
- N层楼，请你设计某种策略确定鸡蛋正好在n层楼会破，鸡蛋在每层楼破的概率都是1/2，其中最好的策略平均最少需要尝试多少次才能一定确定楼层n，假设可使用的鸡蛋为k个。
方法1:
系统的状态: N层楼用了k个鸡蛋尝试的次数为d[N][k]
状态转移方程(action-state),先试第i层，扔一个鸡蛋，消耗一次机会:
- 在i层楼鸡蛋没破，转化为d[N-i][k]
- 在i层楼鸡蛋破了，转化为d[i-1][k-1]
所以有递推公式:
d[N][k] = min_i(max(d[i-1][k-1], d[N-i][k])) + 1

方法2:
已知最大尝试次数，倒推能确定的楼层数
系统的状态: k个鸡蛋允许尝试次数m，最多可以尝试的楼层d[k][m]
状态转移方程(action-state), 假设我们从最优的某一层扔下第一个鸡蛋，必然只有两种结果：
- 鸡蛋碎了: 损失1个鸡蛋，消耗1次机会。下方能测的最大楼层数为d[k-1][m-1]。
- 鸡蛋没碎: 蛋完好，消耗 1次机会。上方能测的最大楼层数为d[k][m-1]。
d[k][m]= d[k-1][m-1] + d[k][m-1] + 1

复杂度对比 (K个鸡蛋, N层楼):
- 方法1: 时间 O(K*N^2), 空间 O(K*N)
  三重循环, 内层枚举决策 i 的次数随 n 增长: sum_n(K*n) = O(K*N^2)
- 方法2: 时间 O(K*N^(1/K)), 空间 O(K*N^(1/K)) (滚动数组可降到 O(K))
  每步只做 O(K) 工作, 而 m 涨到多大由 d[K][m] ~ m^K/K! >= N 决定, 即 m = O(N^(1/K))
  K=2 时方法2为 O(sqrt(N)): N=100 -> m=14, 比方法1的 ~10^4 快几个数量级
本质区别: 方法1"给定楼层求次数"每个状态要枚举所有 i (O(n) 内层);
         方法2"给定次数求楼层"用组合恒等式消掉了内层枚举, 每步仅 O(K)。

policy 提取与"最优value唯一、最优policy不唯一":
- 方法1 的 policy 是 argmin_i, 需额外用 choice[n][k] 记录取到最优的那个 i。
- 方法2 没有显式 argmin, policy 藏在递推的加法 split 里 (起扔层 = d[k-1][m-1]+1)。
- 两者算出的最优次数(value)必然相同(都是 14), 但 extract 出的 policy 可以不同:
  状态(100层,2蛋)下第一扔在 9~14 层都能保证最坏 14 次, 是个有多个最优解的状态。
  方法1 用 `if cost < d[n][k]` 取最小的最优 i (起扔 9 层);
  方法2 的构造法给出平衡解 (起扔 14 层)。
  -> 这正是 MDP 的通性: 最优 value 唯一, 但并列最优的动作可有多个,
     最终 policy 取决于 argmax/argmin 的 tie-breaking。
'''

import builtins
import functools

def tag_print(func):
    """让被装饰函数内的所有 print 自动加上 [函数名] 前缀。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        original_print = builtins.print

        @functools.wraps(original_print)
        def tagged_print(*pargs, **pkwargs):
            original_print(f"[{func.__name__}]", *pargs, **pkwargs)

        builtins.print = tagged_print
        try:
            return func(*args, **kwargs)
        finally:
            builtins.print = original_print  # 确保恢复，异常也不漏
    return wrapper

@tag_print
def solution1():
    N = 100
    K = 2
    d = [ [0] * (K+1) for _ in range(N+1)]
    choice = [ [0] * (K+1) for _ in range(N+1)]  # choice[n][k] = 取到最优的那个 i (argmin), 即 policy
    rows = len(d)          # 结果为 N+1 = 101
    cols = len(d[0])       # 结果为 K+1 = 3
    print(f"Dimensions: {rows}x{cols}")
    for n in range(0, N+1):
        d[n][1] = n        # base case: 1 个鸡蛋只能从下往上逐层试, 需 n 次
        choice[n][1] = 1 if n >= 1 else 0   # 1 个鸡蛋: 总是扔当前区间最低层

    for n in range(1, N+1):
        for k in range(2, K+1):  # k=1 已由上面的 base case 处理, 从 2 开始
            d[n][k] = float('inf')  # 必须先置为无穷大, 否则 min(0, ...) 恒为 0
            for i in range(1, n+1):
                # d[n-i][k] 一定比d[n][k]先cache过
                cost = max(d[i-1][k-1], d[n-i][k]) + 1
                if cost < d[n][k]:        # 记录 argmin 作为 policy; 若<=则与solution2完全一致，否则policy会有差异(最好情况会不一致)虽然平均值一致。
                    d[n][k] = cost
                    choice[n][k] = i
    print("1个鸡蛋最少需要尝试多少次:", d[N][1])
    print("2个鸡蛋最少需要尝试多少次:", d[N][2])

    # 重建 policy: choice[n][k] 直接给出"剩 n 层、k 蛋时第一颗扔区间内第几层(相对底部)"。
    # 没碎 -> 上方子问题 (n-i, k); 碎了 -> 下方子问题 (i-1, k-1)。
    def probe_seq(n, k, base):
        """返回 (n层区间, k蛋, 底部为base) 时, 蛋一直没碎的扔法序列(绝对楼层)。"""
        drops = []
        while n > 0 and k >= 1:
            i = choice[n][k]            # 相对底部第 i 层
            drops.append(base + i)
            base, n = base + i, n - i   # 没碎: 转入上方 n-i 层
        return drops

    main = probe_seq(N, K, 0)           # 第一颗蛋一直没碎的主路径
    print(f"policy (第一颗蛋依次扔的楼层, 共 {len(main)} 次):", main)
    print("相邻间隔(每次没碎后下次少爬1层):", [main[0]] + [b - a for a, b in zip(main, main[1:])])

    # 第一次就碎了: 退化为子问题 (i-1层, K-1蛋), 在下方区间 [1, i-1] 内继续找。
    first_i = choice[N][K]
    broke = probe_seq(first_i - 1, K - 1, 0)
    print(f"若第一次({first_i}层)就碎: 只剩 {K-1} 蛋, 在下方 [1, {first_i - 1}] 逐层试 ->", broke, "\n")

@tag_print
def solution2():
    K = 2
    m = 0
    max_try = 16 # 随机确定的数
    d = [[0] * (max_try) for _ in range(K+1)]  # 这里就不做优化了，0个鸡蛋的内存也放进去了，虽然可以直接从1个鸡蛋开始做，但是还要处理n-1比较麻烦
    while d[K][m] < 100:
        m += 1
        for k in range(1, K + 1):
            d[k][m] = d[k - 1][m - 1] + d[k][m - 1] + 1
        assert m < max_try, "尝试次数过多, 请调大max_try"
    print("1个鸡蛋最多可以尝试的楼层数:", d[1])
    print("2个鸡蛋最少需要尝试多少次:", m)

    # 重建 policy: 沿"蛋一直没碎"的探测路径, 逐步算出每次扔的楼层。
    # 剩 k 个蛋、剩 j 次机会、已确认安全到 base 层时,
    # 下次扔的偏移 = d[k-1][j-1] + 1 (万一碎了, 下方 d[k-1][j-1] 层能用 k-1 蛋 j-1 次测完)。
    N = 100

    def probe_seq(k, j, base):
        """返回 (k蛋, j次机会, 已安全到base层) 时, 蛋一直没碎的扔法序列。"""
        floor = base
        drops = []
        for jj in range(j, 0, -1):      # 每没碎一次, 机会减 1, 鸡蛋数不变
            floor += d[k - 1][jj - 1] + 1
            drops.append(min(floor, N))
            if floor >= N:
                break
        return drops

    main = probe_seq(K, m, 0)           # 第一颗蛋一直没碎的主路径
    print(f"policy (第一颗蛋依次扔的楼层, 共 {len(main)} 次):", main)
    print("相邻间隔(每次没碎后下次少爬1层):", [main[0]] + [b - a for a, b in zip(main, main[1:])])

    # 第一次就碎了: 退化为子问题 (K-1蛋, m-1次), 在下方区间 [1, main[0]-1] 内继续找。
    first = main[0]
    broke = probe_seq(K - 1, m - 1, 0)
    print(f"若第一次({first}层)就碎: 只剩 {K-1} 蛋, 在下方 [1, {first - 1}] 逐层试 ->", broke)



solution1()
solution2()