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
# 上一轮奖励函数代码（该轮得分: 251.358113）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Stage-weighted: early focus on proximity, late focus on precision landing.
    No leg-contact checks — avoids terminal_success_reward proxy.
    """
    # --- hyperparameters ---
    k_proximity = 5.0          # decay rate for bounded proximity
    w_proximity_early = 3.0    # proximity weight in early training
    w_quality_late = 4.0       # landing quality weight in late training
    w_stability = 0.02         # stability penalty weight (mild, background)

    # thresholds for continuous landing quality factors
    D_near = 0.5               # distance: full quality at 0, zero beyond 0.5
    V_slow = 0.5               # speed: full quality at 0, zero beyond 0.5 m/s
    A_upright = 0.3            # angle: full quality at 0, zero beyond 0.3 rad

    # --- extract state ---
    dx, dy = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    dist = (dx * dx + dy * dy) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5

    # --- 1. bounded proximity reward (dense, always active) ---
    proximity = 1.0 / (1.0 + k_proximity * dist)

    # --- 2. continuous landing quality (no leg contacts, purely geometric) ---
    near_factor = max(0.0, 1.0 - dist / D_near)
    slow_factor = max(0.0, 1.0 - speed / V_slow)
    upright_factor = max(0.0, 1.0 - abs(angle) / A_upright)
    landing_quality = near_factor * slow_factor * upright_factor

    # --- 3. mild stability penalty (angular velocity only, to avoid spinning) ---
    stability_penalty = -w_stability * abs(ang_vel)

    # --- stage weighting ---
    # early: proximity dominates → agent learns to approach target
    # late:  landing quality dominates → agent learns precision touchdown
    early_w = max(0.0, 1.0 - training_progress)
    late_w = training_progress

    progress_signal = early_w * w_proximity_early * proximity
    quality_signal = late_w * w_quality_late * landing_quality

    total_reward = progress_signal + quality_signal + stability_penalty

    components = {
        "proximity": proximity,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "progress_signal": progress_signal,
        "quality_signal": quality_signal,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=251.358113, len=375.700000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_quality | 0.318642 | 0.318642 | 0.583516 | 0.318642 |
| progress_signal | 1.186551 | 1.186551 | 1.000000 | 1.186551 |
| proximity | 0.461166 | 0.461166 | 1.000000 | 0.461166 |
| quality_signal | 0.204100 | 0.204100 | 0.583516 | 0.204100 |
| stability_penalty | -0.002367 | 0.002367 | 0.999999 | -0.002367 |
| total_reward | 1.388284 | 1.388284 | 1.000000 | 1.388284 |
| generated_reward | 1.388284 | 1.388284 | 1.000000 | 1.388284 |
| original_env_reward | -0.160388 | 3.418529 | 1.000000 | -0.160388 |

## Distribution
- score: mean=251.358113, min=217.665196, max=276.969335
- episode_length: mean=375.700000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
| 2 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -98.04 | -98.04 | 0.00 | 72.30 | distance_penalty=-0.005 progress_delta_reward=0.016 soft_landing_bonus=0.009 stability_penalty=-0.009 | new_best |
| 3 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -12.26 | -12.26 | 0.00 | 1000.00 | approach_bonus=2.711 distance_penalty=-0.003 progress_delta_reward=0.002 stability_penalty=-0.005 | new_best |
| 4 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -60.67 | -12.26 | -48.41 | 85.30 | approach_bonus=0.005 distance_penalty=-0.004 progress_delta_reward=0.013 stability_penalty=-0.007 | no_meaningful_improvement |
| 5 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -110.31 | -12.26 | -98.05 | 72.00 | approach_bonus=0.009 distance_penalty=-0.005 progress_delta_reward=0.016 stability_penalty=-0.008 | no_meaningful_improvement |
| 6 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -55.92 | -12.26 | -43.65 | 955.10 | approach_bonus=3.129 distance_penalty=-0.002 progress_delta_reward=0.002 stability_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 7 | potential_shaping + soft_landing_proxy + stability_penalty | -108.75 | -108.75 | 0.00 | 72.00 | potential_shaping=0.021 soft_landing_proxy=0.010 stability_penalty=-0.031 | new_best |
| 8 | potential_shaping + soft_landing_proxy + stability_penalty | -0.83 | -0.83 | 0.00 | 1000.00 | potential_shaping=0.005 soft_landing_proxy=0.917 stability_penalty=-0.015 | new_best |
| 9 | landing_quality + progress_signal + proximity + quality_signal + stability_penalty | 251.36 | 251.36 | 0.00 | 375.70 | landing_quality=0.319 progress_signal=1.187 proximity=0.461 quality_signal=0.204 stability_penalty=-0.002 | target_solved_new_best |

```
