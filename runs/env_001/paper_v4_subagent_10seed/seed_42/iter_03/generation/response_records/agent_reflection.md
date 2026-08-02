# Response Record

# 设计理由

## 重建模式：全新骨架方向

### 为什么之前都失败了

**Iter 1**（score 163.33, len 436.40）是唯一成功的轮次——线性进度奖励 + 软着陆引导 + 微弱姿态约束能够让 agent 学会接近并尝试着陆。这是一个可行的基础。

**Iter 2**（score -30.30, len 95.85）在 iter 1 基础上将进度奖励凸化为 `progress + 2·progress²`，期望“刺激 agent 更快到达 proximity 区域”，但实际造成**灾难性崩溃**：len 从 436 暴跌至 96，全部 episode 提前终止（terminated=20/20），其中 45% 是早期灾难性失败（score←50）。因果链清晰——**无约束的凸化进度奖励使 agent 优先追求大步幅接近，失速冲入终止区域（crash 或出界），且没有任何信号在接近危险速度时加以抑制**。

核心矛盾：**progress 组件在奖励“接近目标”，但完全无视“以多快的速度接近”——高速接近后 agent 无法及时减速，必然 crash。**

### 新骨架选了什么算子、和已尝试过的本质不同

**主信号框架：`improvement_delta`（3.2）× `soft_health_gate`（3.5）**

- **Progress 回归线性**（`max(0, dist_prev - dist_next)`），不加凸化。iter 1 已经证明线性可以工作，凸化是毁灭性的。
- **新增 speed safety gate** 乘到 progress 上：当前速度与“安全速度”（正比于距离）比较，超出部分越大，gate 越趋近于 0。结果是——agent 只有**安全地接近**（速度与距离匹配）才能获得进度奖励；盲目冲刺会在 gate 衰减下几乎得不到奖励。
- 这是 **`improvement_delta`（提供梯度方向）× `soft_health_gate`（约束行为边界）** 的显式耦合，不同于 iter 1/2 中 progress 与其他约束组件的**独立求和**——在独立求和中，progress 可以在 agent 即将 crash 时仍然发出强正信号，gate 消除了这种可能性。
- **姿态约束大幅增强**：quadratic penalty 系数从 0.01/0.005 提升至 0.1/0.05，使约束从“存在即可”变为有实质影响力。
- **软着陆引导保留但放宽**：proximity threshold 从 0.2 → 0.3 提升 active_rate；新增 angle_factor 使着陆奖励对姿态更敏感。

**与已尝试路径的本质区别**：新骨架把“速度安全”从**独立约束**（iter 1/2 中几乎没有速度约束）提升为**主信号内部的门控机制**——progress 本身被速度条件化。这直接/精准地打击 iter 2 的崩溃原因，同时保留 iter 1 已验证有效的线性进展方向。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    REBUILD: Safe-progress skeleton with speed-gated advancement.

    Core idea: progress toward target is only rewarded when approach speed is
    proportional to distance.  This prevents the rush→crash failure mode seen
    in iter 2 while preserving the linear progress gradient that worked in iter 1.

    Components:
      1. safe_progress = progress * speed_gate   (main driver)
      2. orientation_penalty                      (stability – strengthened)
      3. soft_landing                             (success proxy – widened)
    """

    # ── Unpack observations ──────────────────────────────────────────
    px0, py0 = obs[0], obs[1]          # last position
    px1, py1 = next_obs[0], next_obs[1]  # current position
    vx1, vy1 = next_obs[2], next_obs[3]  # current velocity
    angle1  = next_obs[4]                # body angle
    angvel1 = next_obs[5]                # angular velocity
    left_leg  = next_obs[6]              # left contact
    right_leg = next_obs[7]              # right contact

    # ── Derived signals ──────────────────────────────────────────────
    dist_prev  = (px0**2 + py0**2) ** 0.5
    dist_next  = (px1**2 + py1**2) ** 0.5
    speed      = (vx1**2 + vy1**2) ** 0.5

    # ── 1. Safe progress (speed-gated advancement) ───────────────────
    # Linear progress – no convexity (the convexity in iter 2 caused the crash)
    raw_progress = dist_prev - dist_next   # positive when approaching
    progress     = max(0.0, raw_progress)

    # Speed safety gate: expected speed ≈ k_target * dist
    # When the agent moves faster than expected for its distance, the gate
    # decays, cutting the reward for reckless approach.
    k_target      = 1.5                    # expected speed / distance ratio
    gate_strength = 3.0                    # how sharply excess speed is penalised

    expected_speed = k_target * dist_next
    excess_speed   = max(0.0, speed - expected_speed)
    speed_gate     = 1.0 / (1.0 + gate_strength * excess_speed**2)

    safe_progress  = progress * speed_gate

    # ── 2. Orientation / stability penalties (strengthened) ──────────
    # Coefficients raised 10× from iter 1/2 so these constraints actually bite.
    angle_penalty  = -0.1 * (angle1 ** 2)
    angvel_penalty = -0.05 * (angvel1 ** 2)
    orientation_penalty = angle_penalty + angvel_penalty

    # ── 3. Soft landing guidance (widened proximity, added angle factor) ──
    proximity_threshold = 0.3             # widened from 0.2 to raise active_rate
    if dist_next < proximity_threshold:
        contact_factor = (left_leg + right_leg) / 2.0    # ∈ [0, 1]
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)       # decays with speed
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle1**2))  # decays with tilt
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ──────────────────────────────────────────────────────
    total_reward = (
        1.0 * safe_progress
        + 1.0 * orientation_penalty
        + 2.0 * soft_landing          # amplify success proxy relative to per-step progress
    )

    components = {
        "safe_progress":       safe_progress,
        "orientation_penalty": orientation_penalty,
        "soft_landing":        soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要

- **audit**：所有观测维度已被使用，但 iter 2 崩溃的根因不是信号缺失，而是**主进展信号的结构缺陷**——progress 奖励了“接近”却无视“有多快接近”，导致无约束冲刺。新骨架通过 speed gate 将速度安全直接嵌入主信号内部，弥补了 iter 1/2 共有的结构缺陷。
- **behavior**：iter 2 中 agent 学会了为获取凸化 progress 奖励而高速冲向目标，在 ~96 步内失去控制并触发 crash/出界终止，无法到达 proximity 区域触发 soft_landing。
- **signal**：speed 信号（obs[2], obs[3]）在 iter 1/2 中仅在 soft_landing 的 <0.2 范围内使用（active_rate 仅 3.4%），对 agent 整体行为**几乎无约束力**。speed gate 将其推广为全时段的 progress 调制器，使得速度在全局尺度上影响奖励——这是缺失的关键结构。
- **level**：Level 3（重建骨架）
- **hypothesis**：speed gate 切断了 agent“冲得越快奖励越多”的正反馈路径——当速度超出与距离匹配的安全阈值时，progress 奖励被大幅削减，迫使 agent 以受控速度接近目标。这应恢复 iter 1 水平的存活时长（len ~400+），同时改善终端着陆质量（soft_landing active_rate 上升）。
- **risk**：如果 `k_target=1.5` 过低（即预期的“安全速度”太保守），agent 可能过度谨慎、进步缓慢，len 过长但 score 停滞。后续可通过调节 `k_target` 调整激进/保守平衡。
