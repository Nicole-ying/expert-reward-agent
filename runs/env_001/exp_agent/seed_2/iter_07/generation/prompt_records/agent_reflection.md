# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你可以调用工具来搜索技法、查看骨架细节。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库。当你对某个症状不确定怎么修时，用自然语言搜索。如 "reward sparse trigger rate low"、"penalty dominating progress"、"oscillation near goal"。
- get_skeleton_detail(skeleton_name)：查看一个骨架的数学形态、原理和陷阱。当你想尝试不同的骨架时，先查看它的细节再决定。

# 怎么修订

你的工具箱里有三种层次的操作：

**层次 1：改系数。** ratio_to_progress 帮助你判断组件间的相对量级，但判断前先看外部得分。
- 惩罚项（stability、energy）ratio 绝对值 > 0.5，且外部得分差 → 考虑削弱或距离门控。
- bonus/proxy（soft_landing）的 ratio 天然偏高——它只在特定条件触发，均值对比的是全时段 progress。只要 nonzero_rate > 10% 且外部得分不差，即使 ratio 很大也不需要修。
- nonzero_rate < 2% 的正向组件 → 增大权重或放宽条件。

**层次 2：改表达式。** 如果系数调了几轮还是不行，考虑改变数学形式。例如：
- 二值 if 条件 → 连续乘积（每项用 max(0, 1-x/threshold)）
- 线性惩罚 → bounded 饱和（1/(1+kx)、tanh、exp(-x)）
- 全程生效 → 距离门控（只在靠近目标时生效）

**层次 3：换骨架 (rebuild)。** 以下情况必须 rebuild，不要继续在层次 1/2 上打转：
- 同一骨架家族已迭代 2 轮以上，且当前最佳得分仍未超过 target 的 25%（如 target=200，最佳仍在 -100 以下）。
- 或者你已经在这个骨架上改过表达式（层次 2）至少一次，但得分没有实质性改善。

rebuild 不是"换个 landing 的写法"——是换整个主信号框架。例如 progress_delta 换成 potential_based_shaping 或 bounded_proximity。数学形态完全不同。

# 奖励函数迭代的通用原则

以下原则来自大量实验，与环境无关。

## 原则 1：比率是通用语言

不关心组件系数的绝对值。关心组件之间的相对大小：
- 主学习信号应该是最强的正向力。
- 约束/惩罚应该是弱背景信号——如果它的均值超过主信号的 50%，agent 会选择"不动"来避免惩罚，而非"行动"来获取奖励。
- 终端/事件型奖励（偶尔触发）的比率天然偏大，这不代表它有问题。只要触发率正常、外部得分不差，高比率不是 bug。

你从 feedback 的 ratio_to_progress 列能直接读到这些比率。

## 原则 2：数学形态决定梯度质量

- 二值条件 → 无梯度，触发率极低时等于摆设。
- 连续乘积 → 有梯度，但多个因子相乘会使整体信号很弱（每个因子 < 1）。
- 连续加权和 → 每个因子独立贡献梯度，但各因子之间没有"同时满足"的约束。
- bounded 函数（1/(1+kx)、max(0,1-x/D)、tanh）→ 自动限制值域，不受环境尺度影响。
- 距离门控 → 让约束只在相关区域内生效，避免远处干扰探索。

改变一个组件的数学形态时，它的理想系数范围也会随之改变——这是预期的，不需要同时调整其他组件来"平衡"。

## 原则 3：每次改一个信号，让下一轮反馈可归因

如果你同时改了三个组件的系数，下一轮分数变了，你不知道是哪个改动造成的。如果一个改动有用、一个有害、一个无关，它们互相掩盖，你需要多轮才能拆开——每一轮都很昂贵。

建议：每次迭代聚焦一个信号（一个组件的系数，或一个表达式的形式）。换骨架 (rebuild) 是例外——它天然是一次大的方向调整。

## 原则 4：信号之间有天然的耦合，但不要主动同时调

如果你把一个组件的数学形态改了（如二值→连续），它的系数自然需要相应调整——这不叫"同时改两个地方"，叫"改一个组件"。但不要乘机顺手也调其他组件的系数。让下一轮反馈单独告诉你这个形态改动是否有效。

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
# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 计算到目标(0,0)的距离变化
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # 正数表示更接近目标
    progress_delta_reward = 10.0 * progress_delta  # 权重10，鼓励每一步都更接近

    # ========== 稳定/安全约束：stability_penalty（已大幅削弱 + 距离门控） ==========
    # 诊断：上一轮 stability_penalty ratio=-3.45，严重主导，agent 不敢动
    # 修复1：系数降低10倍（从0.5/0.3/0.2 → 0.05/0.03/0.02）
    # 修复2：加入距离门控，只在靠近目标（dist<2.0）时生效，远处不约束
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = abs(next_obs[4])  # 姿态角偏离0的绝对值
    angular_vel_penalty = abs(next_obs[5])  # 角速度绝对值
    
    # 距离门控因子：dist<2.0 时逐渐生效，dist>=2.0 时完全关闭
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # 大幅降低系数（10倍），并应用距离门控
    stability_penalty = -distance_gate * (0.05 * speed + 0.03 * angle_penalty + 0.02 * angular_vel_penalty)

    # ========== 任务完成 proxy：soft_landing_proxy（本轮不修改） ==========
    # 当飞行器接近目标、速度低、姿态稳定且两个支撑接触时给予小奖励
    near_target = (next_dist < 0.5)
    low_speed = (speed < 0.3)
    stable_angle = (abs(next_obs[4]) < 0.2)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0

    # ========== 动作代价：energy_penalty（小权重，本轮不修改） ==========
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5
    elif action == 2:
        engine_use = 1.0
    energy_penalty = -0.1 * engine_use

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_bonus + energy_penalty

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta  # 正奖励表示更接近目标

    # 稳定约束：stability_penalty
    # 惩罚高速、大姿态角和大角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.5 * abs(next_obs[4])  # 姿态角惩罚
    angular_vel_penalty = 0.2 * abs(next_obs[5])  # 角速度惩罚
    speed_penalty = 0.1 * speed  # 速度惩罚
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 任务完成 proxy：soft_landing_proxy
    # 当接近目标、低速、小姿态角且双接触时给予小奖励
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 2.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 动作代价：energy_penalty（小权重）
    # 惩罚使用引擎的动作（action 1,2,3）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    # 构建组件字典
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈
# Training Feedback

## Training outcome
score=-116.949772, len=71.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005432 | 0.005432 | 0.099739 | -0.033587 |
| progress_delta_reward | 0.161717 | 0.170881 | 0.999994 | 1.000000 |
| soft_landing_bonus | 0.010848 | 0.010848 | 0.005424 | 0.067079 |
| stability_penalty | -0.033476 | 0.033476 | 1.000000 | -0.207002 |
| total_reward | 0.133657 | 0.152343 | 1.000000 | 0.826490 |
| generated_reward | 0.133657 | 0.152343 | 1.000000 | 0.826490 |
| original_env_reward | -1.658587 | 2.353407 | 1.000000 | -10.256135 |

## Distribution
- score: mean=-116.949772, min=-143.017581, max=-103.429116
- episode_length: mean=71.800000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -108.81 | -108.81 | 0.00 | 72.00 | energy_penalty=-0.007 landing_bonus=0.013 progress_reward=0.161 stability_penalty=-0.129 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.34 | -108.81 | -1.52 | 72.00 | energy_penalty=-0.006 landing_bonus=0.018 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -115.02 | -108.81 | -6.21 | 71.90 | energy_penalty=-0.006 landing_bonus=0.043 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
| 4 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -227.71 | -108.81 | -118.89 | 697.20 | energy_penalty=-0.044 landing_bonus=2.333 progress_reward=0.009 stability_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 5 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -111.68 | -108.81 | -2.86 | 71.90 | energy_penalty=-0.008 progress_delta_reward=0.161 soft_landing_bonus=0.011 stability_penalty=-0.556 | no_meaningful_improvement |
| 6 | energy_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -116.95 | -108.81 | -8.14 | 71.80 | energy_penalty=-0.005 progress_delta_reward=0.162 soft_landing_bonus=0.011 stability_penalty=-0.033 | no_meaningful_improvement |

```
