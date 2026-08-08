# 设计理由

## 信号覆盖审计

- **终止模式**：terminated=20/20，truncated=0，多数 episode 因 `body_fallen_over` 终止；score 负值但 reward 组件合计为正（gated_forward_sum≈98，penalty≈-0.86），说明当前奖励函数与实际任务评分**严重脱节**——model 很可能在“向前摔倒前几步获得高正奖励，但真实环境给出大负分”。
- **观测使用**：仅使用了 `obs[0]`（hull_angle）和 `obs[2]`（horizontal_speed）。未使用的关键信号包括：
  - `obs[1]` **hull_angular_velocity** —— 直接反映机体倾翻速率，可提前数十步预判摔倒；
  - `obs[12]/[13]` leg ground contact —— 可用于检测双脚离地的危险状态。
- **信号缺口**：**信号缺失**——角速度未参与决策，导致 gate 只能在倾角已经较大时才会衰减，完全无法阻止“小倾角 + 大角速度”这种即将摔倒但仍能骗到 forward reward 的漏洞。
- **僵尸组件**：无。

## 行为诊断

agent 学会了通过**快速向前倾倒**来最大化 `horizontal_speed`，因为只要当前倾角尚未超过 $0.25$，gate 完全开放，倾倒时的水平速度仍能带来大量正奖励；等到倾角进入衰减区或 agent 实际摔倒时 episode 已经结束，惩罚来得太晚。这解释了为什么组件合计正收益但最终 score 为负。

## 干预层级：**Level 2 — 结构变换**

方向：在 **gate 计算中新增基于 `hull_angular_velocity` 的衰减因子**，使 gate 能够**提前对快速倾翻做出反应**。同时保留原有 tilt hinge penalty 作为硬边界，但对 gate 单组件进行变换。

公式设计要点：
- 保持原有 tilt gate 逻辑不变。
- 新增 `angvel_gate`：`abs(angvel) ≤ 0.5` → gate=1；`≥ 1.0` → gate=0；中间线性。
- 最终 `gate = tilt_gate × angvel_gate`。
- 系数未动，因此当角速度正常时行为与上一轮完全一致，不会干扰已学会的稳定部分；当角速度异常时提前关断 forward reward，根除 reward hack。

**设计校准**：
- 角速度阈值 `SAFE_ANGVEL=0.5`/`CRITICAL=1.0` 较为宽松，正常行走的 `hull_angular_velocity` 通常 <0.3，不会误压 gate。
- 惩罚负担 <0.3×主信号，仍极低；主改变是 gate 的衰减，不会引入新的大惩罚。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Bipedal locomotion reward for rough terrain:
    - Primary: forward velocity reward with a compound health gate
      that depends on both hull angle and hull angular velocity.
    - Constraint: hinge tilt penalty (unchanged).
    """

    # ==================== Extract signals ====================
    next_hull_angle   = next_obs[0]
    next_hull_angvel  = next_obs[1]                     # newly used
    horizontal_speed  = next_obs[2]

    # ==================== Constants ====================
    # Tilt gate thresholds (unchanged)
    TILT_CRITICAL      = 0.6
    TILT_WARNING_START = 0.25
    TILT_WARNING_MARGIN = 0.35

    # New angvel gate thresholds
    SAFE_ANGVEL    = 0.5
    CRITICAL_ANGVEL = 1.0

    # Hinge penalty constants (unchanged)
    SAFE_TILT   = 0.3
    TILT_WEIGHT = 0.5

    # Forward weight (unchanged)
    FORWARD_WEIGHT = 2.0

    # ==================== Compound gate ====================
    # -- Tilt factor (identical to previous version) --
    abs_tilt = abs(next_hull_angle)
    if abs_tilt <= TILT_WARNING_START:
        tilt_gate = 1.0
    elif abs_tilt >= TILT_CRITICAL:
        tilt_gate = 0.0
    else:
        tilt_gate = (TILT_CRITICAL - abs_tilt) / TILT_WARNING_MARGIN

    # -- Angular velocity factor (new) --
    abs_angvel = abs(next_hull_angvel)
    if abs_angvel <= SAFE_ANGVEL:
        angvel_gate = 1.0
    elif abs_angvel >= CRITICAL_ANGVEL:
        angvel_gate = 0.0
    else:
        angvel_gate = (CRITICAL_ANGVEL - abs_angvel) / (CRITICAL_ANGVEL - SAFE_ANGVEL)

    gate = tilt_gate * angvel_gate

    # ==================== Component A: Gated forward progress ====================
    forward_reward  = FORWARD_WEIGHT * horizontal_speed ** 2
    gated_forward   = gate * forward_reward

    # ==================== Component B: Hinge tilt penalty (unchanged) ====================
    tilt_excess   = max(0.0, abs_tilt - SAFE_TILT)
    tilt_penalty  = TILT_WEIGHT * tilt_excess
    stability_hinge_penalty = -tilt_penalty

    # ==================== Total reward ====================
    total_reward = gated_forward + stability_hinge_penalty

    # ==================== Components dict ====================
    components = {
        'gated_forward_speed': gated_forward,
        'stability_tilt_hinge_penalty': stability_hinge_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺失——未使用 hull_angular_velocity 导致无法预判快速倾倒，当前 reward hack 是 exploit“小倾角 + 大角速度”仍能获得全额 forward reward。
- **behavior**: agent 用向前摔倒的方式骗取高速正奖励，rewards 组件总和为正但真实环境 score 为负。
- **signal**: 缺少角速度预判，gate 反应迟钝；未使用 obs[1]。
- **level**: Level 2
- **hypothesis**: 加入角速度 gate 后，即将摔倒时的 forward reward 会被提前关断，reward hack 失效，迫使 agent 学习真正稳定的步态。
- **risk**: 若角速度阈值过紧可能抑制正常步态的 reward，但当前 SAFE_ANGVEL=0.5 宽松设置下副作用很小；极保守的 cautious 步态可能略微降低前进速度，但仍在可接受范围。