# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你可以调用工具来搜索技法、查看骨架细节。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库。当你对某个症状不确定怎么修时，用自然语言搜索。如 "reward sparse trigger rate low"、"penalty dominating progress"、"oscillation near goal"。
- get_skeleton_detail(skeleton_name)：查看一个骨架的数学形态、原理和陷阱。当你想尝试不同的骨架时，先查看它的细节再决定。

# 怎么修订

你的工具箱里有三种层次的操作：

**层次 1：改系数。** 如果问题是量级不对（如惩罚太强），调系数。用 ratio_to_progress 判断：绝对值 > 0.5 且外部得分差 → 削 10 倍或改为距离门控。nonzero_rate < 2% 的正向组件 → 增大权重或放宽条件。

**层次 2：改表达式。** 如果系数调了几轮还是不行，考虑改变数学形式。例如：
- 二值 if 条件 → 连续乘积（每项用 max(0, 1-x/threshold)）
- 线性惩罚 → bounded 饱和（1/(1+kx)、tanh、exp(-x)）
- 全程生效 → 距离门控（只在靠近目标时生效）

**层次 3：换骨架 (rebuild)。** 如果当前骨架已经调了 2 轮以上、得分仍然远低于 target，不要继续在同一个骨架上微调。从 expert knowledge 中选一个数学形态不同的骨架重来。例如 progress_delta 不行 → 试试 bounded_proximity 或 potential_based_shaping。换骨架之前先调用 get_skeleton_detail 了解候选骨架。

# 改动的可检验性

尽量每次只修改一个方面（一个系数、一个表达式、一个组件），这样下一轮训练后你可以从 ratio_to_progress 的变化判断这个改动是否有效。同时改多个组件会让你无法归因——你不知道哪个改动起作用了。

这不意味着只能改一个数字。把二值改成连续乘积算"改一个组件"。把一个组件的系数调大或调小算"改一个组件"。换骨架 (rebuild) 算"改一个方面"。

# 约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 禁止发明未声明的 info 字段，禁止 import/eval/open。

# 输出

先写注释说明诊断和修改理由，再输出完整 Python 代码。
函数签名：def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
返回 (float(total_reward), components)，components 只放总公式中直接出现的变量。

```

## User Prompt

```markdown
# 环境契约
- function_signature: def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
- allowed: obs[0..7], next_obs[0..7], action (0=noop,1=left,2=main,3=right), info (no reliable fields)
- forbidden: original_reward, official_reward, terminal_success_reward, terminal_failure_penalty


# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward - 奖励每一步更接近目标
    # 计算当前位置到目标(0,0)的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励持续接近
    
    # 稳定约束：stability_penalty - 惩罚高速、大姿态角和角速度
    # 使用next_obs的状态，因为动作影响后的状态更相关
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：平方形式，对高速更敏感
    speed_penalty = -0.5 * (x_vel ** 2 + y_vel ** 2)
    # 姿态角惩罚：角度越大惩罚越大，使用平方
    angle_penalty = -0.3 * (body_angle ** 2)
    # 角速度惩罚：旋转越快惩罚越大
    angular_penalty = -0.2 * (angular_vel ** 2)
    
    stability_penalty = speed_penalty + angle_penalty + angular_penalty
    
    # 任务完成proxy：soft_landing_proxy - 当接近目标且稳定时给予小奖励
    # 条件：距离<0.5，速度<0.3，角度<0.2，角速度<0.2，且至少一个支撑接触
    near_target = next_dist < 0.5
    low_speed = (x_vel ** 2 + y_vel ** 2) ** 0.5 < 0.3
    stable_angle = abs(body_angle) < 0.2
    low_angular_vel = abs(angular_vel) < 0.2
    has_contact = (next_obs[6] > 0.5) or (next_obs[7] > 0.5)
    
    if near_target and low_speed and stable_angle and low_angular_vel and has_contact:
        soft_landing_bonus = 2.0
    else:
        soft_landing_bonus = 0.0
    
    # 动作代价：energy_penalty - 小权重惩罚使用引擎
    # 动作1、2、3都使用引擎，动作0不使用
    if action == 0:
        energy_penalty = 0.0
    else:
        energy_penalty = -0.1  # 小权重，避免agent不敢动
    
    # 总奖励
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty
    
    # 组件字典
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=206.102192, len=498.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.061315 | 0.061315 | 0.613154 | -1.628091 |
| progress_delta_reward | 0.037661 | 0.040800 | 0.996732 | 1.000000 |
| soft_landing_bonus | 0.973998 | 0.973998 | 0.486999 | 25.862305 |
| stability_penalty | -0.116910 | 0.116910 | 1.000000 | -3.104271 |
| total_reward | 0.833433 | 1.071198 | 1.000000 | 22.129943 |
| generated_reward | 0.833433 | 1.071198 | 1.000000 | 22.129943 |
| original_env_reward | -0.045767 | 1.618534 | 1.000000 | -1.215226 |

## Distribution
- score: mean=206.102192, min=112.398817, max=251.917877
- episode_length: mean=498.900000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | 206.10 | 206.10 | 0.00 | 498.90 | energy_penalty=-0.061 progress_delta_reward=0.038 soft_landing_bonus=0.974 stability_penalty=-0.117 | target_solved_new_best |

```
