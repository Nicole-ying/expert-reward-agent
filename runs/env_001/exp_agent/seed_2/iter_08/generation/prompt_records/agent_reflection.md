# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的任务是先理解为什么当前的奖励函数失败了，再决定怎么修改。

# 先诊断，再行动

拿到训练反馈后，先回答三个问题：
1. 这个 agent 发生了什么？episode 很短（<150）→ 在 crash。episode 很长但得分差 → 在徘徊。得分已经好但某组件 ratio 异常 → 可能在 exploit。
2. 哪个组件是主要原因？不要只看 ratio，结合 nonzero_rate 和 episode_length 一起判断。
3. 我之前改了什么？从 Agent Memory 看上一轮的动作和效果。如果上次改了 A 但得分没变，这次不要再改 A。

如果你不确定根因，用 search_reward_design_knowledge 查类似的失败模式。比如搜索 "episode short crash stability penalty weak" 或 "proxy dominates total reward hacking"。

# 工具

- search_reward_design_knowledge(query)：搜索设计技法库和失败模式库。当你对某个症状不确定原因或不知道怎么修时调用。
- get_skeleton_detail(skeleton_name)：查看某个骨架的数学形态、原理和陷阱。

# 怎么修订

三种层次，从轻到重：

**层次 1：改系数。** ratio_to_progress 判断组件间量级。惩罚项 ratio 绝对值 > 0.5 且外部得分差 → 考虑削弱。bonus 类 ratio 天然偏高，只要 nonzero_rate 正常且得分不差就不用修。nonzero_rate < 2% → 增大权重或放宽条件。

**层次 2：改数学形式。** 同一个系数反复调还是不行，说明当前数学形式本身有问题。考虑改变信号的计算方式——但每次只改一个组件的形式，下一轮看效果。

**层次 3：换骨架。** 以下情况停止在层次 1/2 上打转，直接换主信号框架：
- 同一骨架家族已迭代 2 轮以上，且最佳得分仍未超过 target 的 25%。
- 或者已经改过数学形式（层次 2）但得分没有实质性改善。

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
# 上一轮奖励函数代码（该轮得分: -37.739748）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 诊断与修改说明 ==========
    # 问题：6轮迭代，最佳得分-108.81，当前-116.95，所有episode提前终止。
    # 核心问题：progress_delta 信号不足以引导agent到达目标区域。
    # 
    # 修改1（换骨架）：用 bounded_proximity_reward 替代 progress_delta_reward
    #   - 数学形态：1/(1+k*dist)，自动 bounded 在 [0,1]
    #   - 靠近目标时信号饱和，不会因距离变化剧烈而震荡
    #   - k=5 使初始距离(~5)处 reward≈0.17，与当前 progress_delta 均值相当
    #
    # 修改2（换骨架）：用 potential_based_shaping 补充梯度
    #   - Φ = -distance - 0.3*speed - 0.2*abs(angle)
    #   - 势能差 F = γ*Φ(next) - Φ(current)，γ=0.99
    #   - 同时引导接近、减速、稳定，且保持最优策略不变
    #
    # 修改3（层次2）：soft_landing_bonus 从二值改为连续乘积
    #   - 每个条件用 max(0, 1-dim/threshold) 形式
    #   - 提供密集梯度，不再依赖<1%的触发率
    #
    # 修改4（层次1）：stability_penalty 进一步削弱并改为 bounded 形式
    #   - 用 1/(1+k*x) 形式使惩罚有上界
    #   - 系数从 0.05/0.03/0.02 降到 0.02/0.01/0.01
    #   - 保持距离门控
    
    # ========== 公共计算 ==========
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    current_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    next_speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    next_angle = abs(next_obs[4])
    next_angular_vel = abs(next_obs[5])
    
    # ========== 主信号1：bounded_proximity_reward ==========
    # 1/(1+k*dist)，自动 bounded 在 [0,1]，k=5
    k_proximity = 5.0
    proximity_reward = 1.0 / (1.0 + k_proximity * next_dist)
    # 放大到与 progress_delta 相当的尺度
    proximity_reward_scaled = 10.0 * proximity_reward
    
    # ========== 主信号2：potential_based_shaping ==========
    # Φ = -distance - 0.3*speed - 0.2*abs(angle)
    # F = γ*Φ(next) - Φ(current)
    gamma = 0.99
    phi_current = -(current_dist + 0.3 * current_speed + 0.2 * abs(obs[4]))
    phi_next = -(next_dist + 0.3 * next_speed + 0.2 * next_angle)
    shaping_reward = gamma * phi_next - phi_current
    # 放大到合理尺度
    shaping_reward_scaled = 5.0 * shaping_reward
    
    # ========== 稳定约束：bounded_stability_penalty ==========
    # 用 1/(1+k*x) 形式使惩罚 bounded，避免数值爆炸
    # 系数大幅降低，保持距离门控
    gate_radius = 2.0
    distance_gate = max(0.0, 1.0 - next_dist / gate_radius)
    
    # bounded 形式的惩罚项，每项在 [0,1) 范围内
    speed_penalty = 1.0 / (1.0 + 1.0 / (next_speed + 0.01))  # 约等于 1 - exp(-speed)
    # 简化：直接用 bounded 线性形式
    speed_penalty_bounded = next_speed / (next_speed + 1.0)  # bounded in [0,1)
    angle_penalty_bounded = next_angle / (next_angle + 0.5)  # bounded in [0,1)
    angular_vel_bounded = next_angular_vel / (next_angular_vel + 1.0)  # bounded in [0,1)
    
    stability_penalty = -distance_gate * (
        0.02 * speed_penalty_bounded + 
        0.01 * angle_penalty_bounded + 
        0.01 * angular_vel_bounded
    )
    
    # ========== 连续 soft_landing_proxy ==========
    # 用连续乘积替代二值 if 条件
    # 每个因子用 max(0, 1-dim/threshold) 形式
    near_factor = max(0.0, 1.0 - next_dist / 0.5)  # dist<0.5 时>0
    low_speed_factor = max(0.0, 1.0 - next_speed / 0.3)  # speed<0.3 时>0
    stable_angle_factor = max(0.0, 1.0 - next_angle / 0.2)  # angle<0.2 时>0
    # contact 用连续值（obs[6], obs[7] 是接触传感器，假设在 [0,1]）
    contact_factor = next_obs[6] * next_obs[7]  # 两个支撑都接触时接近1
    
    # 连续乘积：所有因子相乘，提供密集梯度
    soft_landing_bonus = 3.0 * near_factor * low_speed_factor * stable_angle_factor * contact_factor
    
    # ========== 动作代价：energy_penalty（小权重） ==========
    engine_use = 0.0
    if action == 1 or action == 3:
        engine_use = 0.5
    elif action == 2:
        engine_use = 1.0
    energy_penalty = -0.05 * engine_use
    
    # ========== 总奖励 ==========
    total_reward = proximity_reward_scaled + shaping_reward_scaled + stability_penalty + soft_landing_bonus + energy_penalty
    
    # ========== 组件字典 ==========
    components = {
        "proximity_reward": proximity_reward_scaled,
        "shaping_reward": shaping_reward_scaled,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-37.739748, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| energy_penalty | -0.029615 | 0.029615 | 0.803217 | -0.029615 |
| proximity_reward | 3.683910 | 3.683910 | 1.000000 | 3.683910 |
| shaping_reward | 0.051027 | 0.056905 | 1.000000 | 0.051027 |
| soft_landing_bonus | 0.326787 | 0.326787 | 0.187213 | 0.326787 |
| stability_penalty | -0.004575 | 0.004575 | 0.999378 | -0.004575 |
| total_reward | 4.027534 | 4.027534 | 1.000000 | 4.027534 |
| generated_reward | 4.027534 | 4.027534 | 1.000000 | 4.027534 |
| original_env_reward | -0.395825 | 2.817910 | 1.000000 | -0.395825 |

## Distribution
- score: mean=-37.739748, min=-76.726194, max=-2.984833
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
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
| 7 | energy_penalty + proximity_reward + shaping_reward + soft_landing_bonus + stability_penalty | -37.74 | -37.74 | 0.00 | 1000.00 | energy_penalty=-0.030 proximity_reward=3.684 shaping_reward=0.051 soft_landing_bonus=0.327 stability_penalty=-0.005 | new_best |

```
