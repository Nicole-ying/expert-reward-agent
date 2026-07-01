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

**层次 3：换骨架 (rebuild)。** 如果当前骨架已经调了 2 轮以上、得分仍然远低于 target，不要继续在同一个骨架上微调。从 expert knowledge 中选一个数学形态不同的骨架重来。例如 progress_delta 不行 → 试试 bounded_proximity 或 potential_based_shaping。换骨架之前先调用 get_skeleton_detail 了解候选骨架。

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
# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: runs\env_001\exp_agent\seed_2\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 当前奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ============================================================
    # 诊断 1：stability_penalty 的 ratio_to_progress = -0.089，已正常 ✅
    # 诊断 2：landing_bonus 的 nonzero_rate = 0.6%，仍然几乎不触发
    #   原因：contact_factor 是二进制门控，只要未接触地面乘积就为 0
    #   修复：去掉 contact_factor，只保留距离、速度、姿态角的连续乘积
    #   这样 agent 在靠近目标的过程中每一步都能感受到梯度
    # ============================================================
    
    # 主学习信号：progress_delta_reward（保持不变）
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta

    # 稳定约束：stability_penalty（系数已降低 10 倍，保持）
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.05 * abs(next_obs[4])
    angular_vel_penalty = 0.02 * abs(next_obs[5])
    speed_penalty = 0.01 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # ============================================================
    # 修复：去掉 contact_factor 的二进制门控
    # 连续乘积只基于距离、速度、姿态角，每个因子用 max(0, 1-x/threshold)
    # 这样 agent 在靠近目标、减速、摆正姿态时都能获得梯度
    # 系数从 5.0 提高到 10.0，因为去掉 contact 后信号更纯粹
    # ============================================================
    
    # 距离因子：距离 < 0.5 时为正，越近越大
    dist_factor = max(0.0, 1.0 - next_dist / 0.5)
    # 速度因子：速度 < 0.5 时为正，越慢越大
    speed_factor = max(0.0, 1.0 - speed / 0.5)
    # 姿态角因子：角度 < 0.3 时为正，越小越大
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)
    
    # 连续乘积：每个因子都在 [0,1]，乘积也在 [0,1]
    # 去掉 contact_factor，让梯度在每一步都能传递
    landing_bonus = 10.0 * dist_factor * speed_factor * angle_factor

    # 动作代价：energy_penalty（保持不变）
    engine_use = 1.0 if action != 0 else 0.0
    energy_penalty = -0.05 * engine_use

    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + energy_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
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
score=-115.024416, len=71.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.005695 | 0.005695 | 0.113908 | -0.035362 |
| landing_bonus | 0.042785 | 0.042785 | 0.007405 | 0.265647 |
| progress_reward | 0.161058 | 0.170415 | 0.999997 | 1.000000 |
| stability_penalty | -0.014429 | 0.014429 | 1.000000 | -0.089590 |
| total_reward | 0.183718 | 0.199111 | 1.000000 | 1.140695 |
| generated_reward | 0.183718 | 0.199111 | 1.000000 | 1.140695 |
| original_env_reward | -1.617069 | 2.341746 | 1.000000 | -10.040299 |

## Distribution
- score: mean=-115.024416, min=-130.746612, max=-98.057243
- episode_length: mean=71.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -108.81 | -108.81 | 0.00 | 72.00 | energy_penalty=-0.007 landing_bonus=0.013 progress_reward=0.161 stability_penalty=-0.129 | new_best |
| 2 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -110.34 | -108.81 | -1.52 | 72.00 | energy_penalty=-0.006 landing_bonus=0.018 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |
| 3 | energy_penalty + landing_bonus + progress_reward + stability_penalty | -115.02 | -108.81 | -6.21 | 71.90 | energy_penalty=-0.006 landing_bonus=0.043 progress_reward=0.161 stability_penalty=-0.014 | no_meaningful_improvement |

```
