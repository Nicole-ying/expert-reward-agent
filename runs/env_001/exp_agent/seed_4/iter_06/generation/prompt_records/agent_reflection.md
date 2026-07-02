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
# 上一轮奖励函数代码（该轮得分: -54.335404）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.01        # 【降低】0.03→0.01，缓解 landing proxy 对 progress 的碾压（ratio 7x→~2.3x）

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

    # --- Soft landing proxy (CONTINUOUS) ---
    # 三个 bounded 因子，每个 ∈ [0, 1]，乘积提供连续梯度
    proximity    = max(0.0, 1.0 - dist_next / D_max)
    smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
    upright      = max(0.0, 1.0 - abs_angle / A_max)

    landing_bonus = w_landing * proximity * smooth_speed * upright

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- HYPERPARAMETERS ----------
    w_progress = 1.0        # 主学习信号：每步接近目标给予奖励
    w_vel     = 0.01        # 速度惩罚权重（水平+垂直）
    w_angle   = 0.01        # 倾斜角度惩罚权重
    w_angvel  = 0.005       # 角速度惩罚权重
    w_landing = 0.03        # 软着陆连续代理奖励（从 0.3 大幅降低，避免劫持 progress）

    # 连续 landing proxy 的阈值（bounded 因子归零点）
    D_max = 2.0             # 距离阈值：超过此距离 proximity 归零
    V_max = 3.0             # 速度阈值：超过此速度 smooth_speed 归零
    A_max = 0.5             # 角度阈值（弧度，~28°）：超过此角度 upright 归零
    # --------------------------------

    # --- Progress toward target ---
    dist_obs  = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress_delta = dist_obs - dist_next
    progress_reward = w_progress * progress_delta

    # --- Stability penalty ---
    vx, vy = next_obs[2], next_obs[3]
    angle  = next_obs[4]
    angvel = next_obs[5]

    abs_v_sum = abs(vx) + abs(vy)
    abs_angle = abs(angle)
    abs_angvel = abs(angvel)

    stability_penalty = -(w_vel * abs_v_sum + w_angle * abs_angle + w_angvel * abs_angvel)

    # --- Soft landing proxy (CONTINUOUS) ---
    # 三个 bounded 因子，每个 ∈ [0, 1]，乘积提供连续梯度
    proximity    = max(0.0, 1.0 - dist_next / D_max)
    smooth_speed = max(0.0, 1.0 - abs_v_sum / V_max)
    upright      = max(0.0, 1.0 - abs_angle / A_max)

    landing_bonus = w_landing * proximity * smooth_speed * upright

    # --- Total reward ---
    total_reward = progress_reward + stability_penalty + landing_bonus

    components = {
        "total_reward": total_reward,
        "progress_delta_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-54.335404, len=76.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.015284 | 0.016240 | 0.999995 | 1.000000 |
| soft_landing_proxy | 0.002613 | 0.002613 | 0.993934 | 0.170947 |
| stability_penalty | -0.013359 | 0.013359 | 1.000000 | -0.874052 |
| total_reward | 0.004538 | 0.008765 | 1.000000 | 0.296895 |
| generated_reward | 0.004538 | 0.008765 | 1.000000 | 0.296895 |
| original_env_reward | -0.971416 | 2.758795 | 1.000000 | -63.557543 |

## Distribution
- score: mean=-54.335404, min=-91.502316, max=-19.793447
- episode_length: mean=76.200000
- early_terminal (<150 steps + score<-50): 6/10 (60%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.33 | -107.33 | 0.00 | 71.50 | progress_delta_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.014 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 149.94 | 149.94 | 0.00 | 1000.00 | progress_delta_reward=0.002 soft_landing_proxy=0.234 stability_penalty=-0.002 | new_best |
| 3 | progress_delta_reward + soft_landing_proxy + stability_penalty | 165.26 | 165.26 | 0.00 | 1000.00 | progress_delta_reward=0.003 soft_landing_proxy=0.024 stability_penalty=-0.003 | new_best |
| 4 | progress_delta_reward + soft_landing_proxy + stability_penalty | 157.71 | 165.26 | -7.55 | 1000.00 | progress_delta_reward=0.003 soft_landing_proxy=0.023 stability_penalty=-0.004 | no_meaningful_improvement |
| 5 | progress_delta_reward + soft_landing_proxy + stability_penalty | -54.34 | 165.26 | -219.60 | 76.20 | progress_delta_reward=0.015 soft_landing_proxy=0.003 stability_penalty=-0.013 | no_meaningful_improvement |

```
