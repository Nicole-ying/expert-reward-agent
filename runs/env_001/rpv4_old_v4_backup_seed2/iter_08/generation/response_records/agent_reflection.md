# Response Record

---

## 1. 诊断报告

### evidence

**跨轮语义追踪（聚焦"着陆"与"下降引导"）：**

| iter | score | len | term | landing 语义 | 机制 | 状态 |
|------|-------|-----|------|-------------|------|------|
| 1 | -203.5 | 185.8 | 20/0 | contact-gated 乘积 | 稀疏 | dead (0%) |
| 2 | -37.0 | 962.5 | 4/16 | proximity-gated 连续和 | 连续 | **exploited** (93.8% share, 悬停 962 步) |
| 3-5 | 负分 | ~200 | 全 crash | 反复切换连续/稀疏 | — | dead |
| 6 | -13.6 | 1000 | 0/20 | contact-gated 乘积稀疏 | 稀疏 | dead (0%)，全截断 |
| **7** | **146.3** | **673.4** | **14/6** | contact-gated 乘积稀疏 + **descent_bonus 桥接** | 稀疏+连续桥 | **突破** (19% active, 70.5% share) |

**iter 7 vs iter 6 的关键变化：** 添加 `descent_bonus`（3.0 × x_gate × descent），使 landing_completion 从 dead→19% active，14/20 成功着陆。

**当前问题：**
- 得分 146.32，目标 200，差距 ~27%
- 6/20 截断（30% 的 episode 无法在 1000 步内着陆）
- score_range [0.90, 252.42]：截断 episode 得分极低，拖累均值
- `descent_bonus` ep_sum 仅 3.28，在 checkpoint 全训练过程几乎恒定（2.5→3.4），说明它只是被动跟踪下降量，**没有主动塑造行为**——agent 并未因 descent_bonus 的存在而学得更快或更陡地下降
- `landing_completion` magnitude_share=70.5%（<90%，未触发 exploit 阈值），但稀疏信号与连续信号的比值约 14:1，连续梯度太弱

**Checkpoint 关键证据：**
- descent_bonus ep_sum 从 cp10% 到 cp100% 始终在 2.5–3.4 窄区间波动——agent 全训练过程下降速率从未改变
- landing_completion ep_sum 在 cp70%→cp80% 从 0 跳到 24.3——这是"偶然发现着陆"的典型稀疏奖励模式
- score 在 cp90% 骤降至 105.3，说明训练不稳定，agent 在探索中有时丢失着陆能力

### behavior_diagnosis

agent 已经学会了基本的着陆能力（14/20），但 30% 的 episode 无法着陆。这些截断 episode 中，agent 很可能在中等高度悬停或水平漂移，无法完成最后的下降。核心原因是 **descent_bonus 太弱（3.0 系数），无法提供足够的梯度来引导 agent 从"接近目标区"过渡到"双脚着陆"**。agent 依赖偶然触发 landing_completion 来学习着陆，而非通过连续的下降梯度主动引导。

### signal_completeness

- `approach_target`（接近目标）：✅ distance_progress 提供全高度欧氏距离减少梯度，工作正常
- `stable_landing`（稳定着陆）：⚠️ landing_completion 提供成功信号但稀疏（19%），descent_bonus 提供桥接但太弱
- `action_penalty`（燃料节省）：✅ fuel_penalty 提供轻微动作成本，不再主导
- **缺失信号**：无需新增信号类型。现有 descent_bonus 的职责覆盖正确，只是力度不足

### selected_level

**L1** — 仅调整系数。`descent_bonus` 的数学形态（x_gate × per-step-descent）是正确的：它奖励"在目标垫上方时每步下降的量"，悬停得 0，上升得 0，不会被 exploit。问题纯粹是系数 3.0 太小，无法与稳定性惩罚和燃料成本竞争，无法驱动行为变化。

### selected_intervention

**目标组件：`descent_bonus`**  
**修改：** 系数从 `3.0` 提升到 `6.0`

不改 x_gate 宽度、不改公式结构、不改其他组件。这是单一、干净的 L1 干预。

### falsifiable_hypothesis

更强的下降奖励梯度将使 agent 在训练中更早、更稳定地学会下降行为。预期效果：
1. descent_bonus ep_sum 将从 ~3.3 升至 ~5-7（agent 下降得更果断）
2. 截断率从 6/20 降至 2-4/20（更多 episode 成功着陆）
3. 得分从 146 升至 ~170-200
4. landing_completion magnitude_share 将从 70.5% 降至 ~50-60%（连续信号占比增大）

### expected_next_round

- `descent_bonus` ep_sum 应增长至 5-8，active_rate 保持在 55-75%（agent 在目标区上方花更多时间下降）
- `landing_completion` ep_sum 可能微增至 50-60（更多 episode 着陆成功）
- `stability` ep_sum 可能变得更负（-4 到 -6），因为 agent 在更低高度有更高速度
- `terminated` 应从 14/20 升至 16-18/20

### main_risk

**下降过快导致 crash。** 如果 descent_bonus 权重翻倍后，agent 学会"俯冲"以最大化下降奖励，可能在接触垫时速度过高而 crash。但现有 stability 组件的速度惩罚（`0.1 × vel² × gate`）在低高度时会显著激活（gate→1），提供自然的刹车信号。如果下一轮出现大量 crash（terminated 增加但 score 下降），则需要同时调整 stability 的 w_vel 系数。

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
      D. landing_completion  – small soft success bonus for two-leg contact with low speed
      E. fuel_penalty        – tiny penalty for using any engine
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
    #    v8 change: coefficient 3.0 → 6.0 to strengthen descent gradient.
    #    The per-step descent amount is naturally small (~0.01–0.05 per step),
    #    and at 3.0 the total per-episode signal (~3.3) was too weak to
    #    compete with other components and actively shape descent behavior.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)          # 1 at x=0, 0 at |x|>=0.5
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 6.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    # =====================================================================
    both_feet = left_contact * right_contact            # 1.0 only if both are 1.0
    close_to_centre = current_distance < 0.3
    if both_feet > 0.5 and close_to_centre:
        speed_quality = max(0.0, 1.0 - velocity_norm / 0.5)
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.5)
        angvel_quality = max(0.0, 1.0 - abs(ang_vel) / 0.5)
        landing_reward = 0.5 * speed_quality * angle_quality * angvel_quality
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
