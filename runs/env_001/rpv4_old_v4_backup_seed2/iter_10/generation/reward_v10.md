Now I have all the data I need. Let me compile a thorough diagnosis.

---

## 1. 诊断报告

### evidence

**跨轮对比（iter 8 → iter 9）：**

| 指标 | iter 8 (best, 199.42) | iter 9 (186.54) | Δ |
|------|----------------------|-----------------|-----|
| score | 199.42 | 186.54 | **-12.88** |
| len | 477.6 | 260.9 | -216.7 |
| term | 17/20 (3 truncated) | 20/20 (0 truncated) | +3 |
| landing_completion ep_sum | 70.98 | 19.17 | **-51.81** |
| landing_completion active | 32.3% | 19.4% | -12.9% |
| descent_bonus ep_sum | 6.38 | 6.86 | +0.48 |
| descent_bonus active | 69.7% | 84.5% | +14.8% |
| stability ep_sum | -1.87 | -1.92 | -0.05 |

**唯一代码差异**：iter 8 → iter 9 仅将 `descent_bonus` 的 `x_gate` 从 `/0.5` 扩到 `/0.8`。

**Checkpoint 关键证据（iter 9）：**
- landing_completion: cp90%=**48.1** → cp100%=**19.8**（最后 10% 训练阶段骤降 58%）
- 这意味着 agent 在训练末期学会了**牺牲着陆质量换取速度**——更短的 episode（len 从 iter 8 的 478 降到 261），更快着陆但质量更差。

**语义历史（防止打转）：**
- iter 7: descent_bonus 系数 3→6（改权重）
- iter 8: descent_bonus 系数保持 6，未改
- iter 9: descent_bonus x_gate 0.5→0.8（改形态）
- landing_completion 的乘法因子 0.5 **自 iter 6 以来从未修改过**

### behavior_diagnosis

Agent 当前行为：**快速但不精确着陆**。20/20 成功终止（解决了 iter 8 的 3 个截断），但着陆质量显著下降。descent_bonus 的放宽 x_gate 让 agent 在水平偏移 0.5~0.8 时也获得下降激励，导致它学会了"先降下去再说"——以较差的角度和速度接触垫面，landing_completion 从 70.98 暴跌到 19.17。

Checkpoint 轨迹揭示核心矛盾：agent 在 cp90% 时尚能获得 48.1 的 landing_completion，但在 cp100% 时将其"优化掉"了——因为它发现更快着陆（更多 descent_bonus + distance_progress 累积）比高质量着陆（sparse landing_completion）更划算。**着陆质量信号太弱，被速度信号淹没。**

### signal_completeness

必要职责：
- ✅ approach_target（distance_progress + descent_bonus）
- ✅ stable_landing（landing_completion）——存在但太弱
- ✅ stability（速度/姿态约束）——存在但力度弱
- ✅ fuel_penalty

缺失：landing_completion 的**相对权重不足**。它是唯一的着陆质量信号，但其 per-step 最大贡献（0.5）远小于累积 descent_bonus（6.86/ep）和 distance_progress（6.26/ep）的总和。在一个 260 步 episode 中，descent_bonus 每步贡献 ~0.026，而 landing_completion 只在接触步触发，且每步最大仅 0.5——即使完美着陆 10 步也只有 5.0，远小于全程累积的 descent_bonus。

### selected_level

**L1**——只调系数，不改数学表达式。landing_completion 的方向正确（接触门控 + 质量乘积），问题纯粹是力度不足。

### selected_intervention

**目标组件**：`landing_completion`

**具体修改**：将乘法因子从 `0.5` 提升到 `2.0`（4× 增强）。这使 landing_completion 的 per-step 最大值从 0.5 变为 2.0，让高质量着陆在奖励结构中具有竞争力。

不改 x_gate——保持 iter 9 的 `/0.8`，因为它在解决截断问题上有效（3→0 truncated）。

### falsifiable_hypothesis

提升 landing_completion 权重后，agent 在训练末期（cp90%→cp100%）不会再"优化掉"着陆质量。因为每步高质量着陆现在值 2.0（而非 0.5），累积 descent_bonus（~6.86/ep）不再能轻易压倒它。Agent 会维持 cp90% 水平的着陆质量到训练结束，landing_completion ep_sum 预计回升到 40-80 区间。

### expected_next_round

| 指标 | 预期变化 | 理由 |
|------|---------|------|
| score | **≥200** | landing_completion 回升 20-60 分，超越 iter 8 |
| landing_completion ep_sum | **40–80**（从 19.2 上升） | 4× 权重使 agent 维持着陆质量 |
| landing_completion active_rate | **25–40%**（从 19.4% 上升） | 更多步维持良好着陆姿态 |
| descent_bonus | 基本持平（6-7） | x_gate 未变 |
| len | 可能略增（280–350） | agent 为更高着陆质量可能放慢着陆 |
| term | 保持 20/20 | x_gate 放宽已解决截断问题 |

### main_risk

Landing_completion 权重过高可能导致 agent **在着陆后故意延长 episode**（不主动终止，维持接触状态以累积 bonus）。缓解因素：环境自然终止条件（body_not_awake_or_settled）会在速度/角速度接近零时触发，agent 无法无限延长。但如果出现此问题，下一轮需加 per-episode 上限或改为 one-time bonus。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D lunar lander-like navigation task.
    Components:
      A. distance_progress   – reduction in Euclidean distance to landing pad
      B. stability           – penalise high velocity and large attitude when close to pad
      C. descent_bonus       – reward vertical descent when horizontally aligned with pad
      D. landing_completion  – soft success bonus for two-leg contact with low speed
      E. fuel_penalty        – tiny penalty for using any engine

    v10 change: landing_completion multiplier 0.5 → 2.0.
    The wider x_gate (/0.8) from v9 solved the truncation problem (0/20 truncated),
    but caused landing quality to be undervalued — landing_completion dropped from
    70.98 (v8) to 19.17 (v9) as the agent learned to rush landings.  Increasing
    the multiplier 4× makes high-quality landings competitive with cumulative
    descent/distance signals, so the agent maintains good landing posture through
    the end of training instead of optimizing it away.
    """
    # --- unpack observations ---
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]

    nx_pos = next_obs[0]
    ny_pos = next_obs[1]

    # --- distances ---
    current_distance = (x_pos**2 + y_pos**2) ** 0.5
    next_distance = (nx_pos**2 + ny_pos**2) ** 0.5
    velocity_norm = (x_vel**2 + y_vel**2) ** 0.5

    # =====================================================================
    # A. distance_progress: reward reduction in Euclidean distance to target
    # =====================================================================
    progress = (current_distance - next_distance) * 5.0   # scaling factor

    # =====================================================================
    # B. stability: smooth penalty for speed and attitude when near the pad
    #    gate: exp(-2 * distance)  →  close to 1 when distance small
    # =====================================================================
    gate = 2.718281828 ** (-2.0 * current_distance)     # approximate exp(-2d)
    w_vel = 0.1
    w_angle = 0.5
    w_angv = 0.5
    stability_penalty = - (w_vel  * velocity_norm**2 +
                           w_angle * angle**2 +
                           w_angv  * ang_vel**2) * gate

    # =====================================================================
    # C. descent_bonus: reward vertical descent when horizontally aligned
    #    with the landing pad.  This provides continuous gradient from
    #    altitude to ground, bridging the gap between distance_progress
    #    (which is isotropic) and landing_completion (which is sparse).
    #
    #    Key design: rewards the *change* in y (delta-y), NOT the absolute
    #    y position.  Hovering at low altitude gives zero reward → no exploit.
    #    Gate ensures reward only activates when agent is above the pad.
    #
    #    x_gate /0.8 (from v9): wider than original /0.5, gives partial
    #    descent incentive to agents moderately offset horizontally,
    #    helping them descend while centering instead of requiring perfect
    #    alignment first.  This eliminated the 3/20 truncated episodes.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.8)          # 1 at x=0, 0 at |x|>=0.8
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 6.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    #
    #    v10 change: multiplier 0.5 → 2.0.  The wider descent gate caused
    #    agents to rush landings, undervaluing landing quality.  This 4×
    #    increase makes high-quality landing posture competitive with the
    #    cumulative descent/distance rewards, preventing the agent from
    #    optimizing away landing quality in late training.
    # =====================================================================
    both_feet = left_contact * right_contact            # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 2.0 * speed_quality * angle_quality * angvel_quality
    else:
        landing_reward = 0.0

    # =====================================================================
    # E. fuel_penalty: minuscule cost for any engine use
    # =====================================================================
    fuel_penalty = -0.01 if action != 0 else 0.0

    total_reward = (progress + stability_penalty + descent_bonus
                    + landing_reward + fuel_penalty)

    components = {
        "distance_progress": progress,
        "stability": stability_penalty,
        "descent_bonus": descent_bonus,
        "landing_completion": landing_reward,
        "fuel_penalty": fuel_penalty
    }

    return float(total_reward), components
```