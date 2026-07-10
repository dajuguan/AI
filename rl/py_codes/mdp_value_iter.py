'''
参考David saylor第三课1:07分的例子
'''

from pprint import pprint


GRID_WIDTH = 4
STEPS = 10
GAMA = 1.0
v =  [[0] * GRID_WIDTH for _ in range(GRID_WIDTH)] 
print("初始值函数:")
pprint(v, width=20)


def deep_clone(matrix):
     return [row[:] for row in matrix]


# v' = max(R(s,a) + γ*∑P(s,s',a)*v(s')) over a

for t in range(STEPS):
    v_new = deep_clone(v)
    for i in range(GRID_WIDTH):
        for j in range(GRID_WIDTH):
            # 计算每个动作的价值
            action_values = []
            for action in ['up', 'down', 'left', 'right']:
                if action == 'up':
                    next_i, next_j = max(i-1, 0), j
                elif action == 'down':
                    next_i, next_j = min(i+1, GRID_WIDTH-1), j
                elif action == 'left':
                    next_i, next_j = i, max(j-1, 0)
                elif action == 'right':
                    next_i, next_j = i, min(j+1, GRID_WIDTH-1)
                # reward衡量的是离goal的距离
                # 只有处于goal的位置reward=0，其他位置reward=-1
                reward = -1
                if (i, j) == (0, 0):  
                    reward = 0
                P_s_s_new_a = 1.0  # 确定性转移
                value = reward + P_s_s_new_a*v[next_i][next_j] * GAMA
                action_values.append(value)

            v_new[i][j] = max(action_values)
        v = v_new
    print(f"Step:{t} 更新后的值函数:")
    pprint(v, width=40)