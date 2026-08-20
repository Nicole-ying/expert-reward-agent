# Response Record

# 设计理由

## 改动内容
**删除 `soft_landing` 组件中的 `contact_ok` 因子。**

## 为什么
第 0 步审计发现：

- **终止模式**: `terminated=20/20, len=68.45`，agent 没有一例存活到超时。episode 长度较 Iter 1 无改善（`68.45 → 68.45`）。结合 `score=-109.32`（极度负分），agent 很可能在远离目标的位置触发 `crash_or_body_contact` 或 `horizontal_position_outside_viewport` 终止。

- **信号使用扫描**: 所有 8 个观测维度都已被使用，不存在完全未接触的观测。

- **僵尸组件**: `soft_landing` 的 `active_rate=0.7%`，即 20 个 episode 中只有约 0.14 个 episode 触发过一次——**几乎从未激活**。但 `soft_landing` 是所有组件中唯一一个在 agent 接近目标时提供正向奖励的组件（`progress_reward` 是 delta 形式，在静止或反复徘徊时为 0）。agent 缺乏来自"接近目标"的可持续正向信号。

- **根因**: `soft_landing` 是四因子乘积 `proximity * angle_ok * vel_ok * contact_ok`。其中 `contact_ok = left_n * right_n` 要求**双足同时着地**——这在飞行阶段恒为 0。乘积中任何因子为 0 则整个组件为 0。这意味着即使 agent 飞到目标正上方且姿态完美，只要还没着陆，`soft_landing` 就完全不提供奖励。但 agent 需要正向信号来引导它靠近目标区域——`progress_reward` 单独不足以维持目标导向行为（它在徘徊时为 0，在远离时变负）。

**方向判断**: 当前方向（加惩罚）已连续 2 轮未能提升 episode 长度，继续加更多惩罚只会让 agent 更加保守/不敢移动。需要修复的是**正向引导信号的可用性**——让 agent 在靠近目标时就能获得奖励，而非等到双足着地。

## 修改内容

移除 `contact_ok` 因子，使 `soft_landing` 变为三因子乘积 `proximity * angle_ok * vel_ok`。

- `proximity` 确保奖励只在 `R_next < 0.5` 时激活（接近目标区域）。
- `angle_ok` 和 `vel_ok` 确保靠近目标时姿态和速度也接近着陆条件。
- 移除 `contact_ok` 后，agent 在飞抵目标上方时即可获得正向强化，引导其学会"飞到目标区域"这个子任务。

## 系数校准

- 现有 `soft_landing` 的 `episode_sum_mean=0.043`，但这是 `active_rate=0.7%` 的极稀疏触发下的累计值。扩大可用窗口后触发率将显著上升。
- `LANDING_WEIGHT=0.2` 保持不变，因为三因子乘积在最佳情况（proximity=1, angle_ok=1, vel_ok=1）下给出 `0.2` 的 per-step 奖励，而 `progress_reward` 每步约 `0.016`。`0.2` 适度高于 progress per-step，有足够吸引力但不完全支配。

## 为什么不是其他修改

- **不加新的惩罚**: len 没有改善，agent 已经因为某种终止条件而持续失败，加惩罚只会让策略更保守。
- **不改 `progress_reward`**: 它 active_rate=100%，是唯一持续提供信号的组件，修改它风险高且当前无明确证据需要改。
- **不改 `angular_velocity_penalty`**: 它 active_rate=0.7%，作为一个惩罚项低活跃度是正常的（只在违规时触发），这符合设计意图，不是 bug。
- **不把乘积改几何平均**: 几何平均 `(proximity * angle_ok * vel_ok) ** (1/3)` 无法解决 `contact_ok` 恒为 0 导致整体为 0 的问题（因为 contact_ok 不在几何平均内仍会在原乘积中塌缩）。根本问题是 contact_ok 在飞行阶段永远为 0，移除它是最直接的修复。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for the 2D lander goal-reaching task.
    Drives the agent to reach the target pad and settle gently.
    v3: removed contact_ok from soft_landing so agent gets positive signal
        when approaching the target zone, not only after both feet touch.
    """
    # ---------- constants ----------
    PROGRESS_WEIGHT = 1.0
    LANDING_WEIGHT = 0.2
    ANGLE_PENALTY_WEIGHT = 0.01
    ANGULAR_VELOCITY_PENALTY_WEIGHT = 0.02

    PROXIMITY_THRESHOLD = 0.5
    ANGLE_THRESHOLD = 0.5
    VELOCITY_THRESHOLD = 0.5
    ANGULAR_VELOCITY_THRESHOLD = 0.5

    # ---------- unpack observations ----------
    x_o, y_o, x_v_o, y_v_o, angle_o, angvel_o, left_o, right_o = tuple(obs)
    x_n, y_n, x_v_n, y_v_n, angle_n, angvel_n, left_n, right_n = tuple(next_obs)

    # ---------- 1) progress to target ----------
    R_obs = (x_o ** 2 + y_o ** 2) ** 0.5
    R_next = (x_n ** 2 + y_n ** 2) ** 0.5
    progress_reward = PROGRESS_WEIGHT * (R_obs - R_next)

    # ---------- 2) soft landing incentive (now without contact requirement) ----------
    proximity = max(0.0, 1.0 - R_next / PROXIMITY_THRESHOLD)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / ANGLE_THRESHOLD)
    vel_ok = max(0.0, 1.0 - (abs(x_v_n) + abs(y_v_n)) / VELOCITY_THRESHOLD)

    soft_landing = LANDING_WEIGHT * proximity * angle_ok * vel_ok

    # ---------- 3) light angular penalty ----------
    angle_penalty = -ANGLE_PENALTY_WEIGHT * (angle_n ** 2)

    # ---------- 4) angular velocity hinge penalty ----------
    angular_velocity_penalty = (
        -ANGULAR_VELOCITY_PENALTY_WEIGHT
        * max(0.0, abs(angvel_n) - ANGULAR_VELOCITY_THRESHOLD)
    )

    # ---------- aggregate ----------
    total_reward = progress_reward + soft_landing + angle_penalty + angular_velocity_penalty

    components = {
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
        "angle_penalty": angle_penalty,
        "angular_velocity_penalty": angular_velocity_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有观测已被使用，无信号缺口；`soft_landing` 因 `contact_ok` 恒零而 active_rate=0.7%，是僵尸组件——agent 缺乏接近目标区的正向引导信号。
- **behavior**: agent 在约 68 步后 crash 终止，episode 长度与 Iter 1 相同，没有改善趋势。
- **signal**: 唯一的接近目标正向奖励 `soft_landing` 因要求双足着地而无法在飞行阶段激活，导致 agent 缺少引导其飞向目标区域的正向信号。
- **level**: Level 2 — 结构变换（移除塌缩因子，释放正向引导信号）。
- **hypothesis**: 移除 `contact_ok` 后，agent 在靠近目标且姿态/速度良好时即可获得正向奖励，学会"飞到目标上方"这个子任务，从而延长存活时间、提高接近目标的频率。
- **risk**: 可能在目标区域附近徘徊而不尝试着陆（因为不再需要接触就能获得奖励）。但 `vel_ok` 和 `angle_ok` 仍要求低速度和小角度，限制了纯徘徊的收益空间。如果下一轮出现"悬停但不着陆"，可通过后续添加着陆专项奖励来解决。
