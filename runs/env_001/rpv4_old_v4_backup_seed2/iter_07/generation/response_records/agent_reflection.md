# Response Record

现在我有了完整的画面。让我整理诊断。

---

## 1. 诊断报告

### evidence

**跨轮全局视图（语义追踪）：**

| iter | score | len | term | landing 语义 | landing 状态 |
|------|-------|-----|------|-------------|-------------|
| 1 | -203.53 | 185.8 | 20/0 | contact-gated 乘积 | dead (0%) |
| 2 | -37.03 | 962.5 | 4/16 | **proximity-gated** sum-of-factors | **exploited (93.8% share, ep_sum=697)** |
| 3 | -222.85 | 209.6 | 20/0 | contact-gated 乘积 | dead (0%) |
| 4 | -295.33 | 138.9 | 20/0 | 移除 landing，换 velocity_guidance | N/A (也失败) |
| 5 | -197.46 | 185.5 | 20/0 | proximity-gated + contact_multiplier | dead (0%) |
| 6 | **-13.57** | 1000.0 | **0/20** | contact-gated both_feet + quality | **dead (0%)** |

**着陆语义已切换 4 次（proximity↔contact），形成"proximity=exploit，contact=dead"的死循环。** 两个方向均已穷尽。

**iter 6 关键数据：**
- `landing_completion`: active_rate=**0.0%**, ep_sum=0.000 — 死组件
- `fuel_penalty`: active_rate=98.9%, signed_share=-53.2%, magnitude_share=53.2% — 全覆盖惩罚，每步 -0.01
- `distance_progress`: ep_sum=6.331 (每步均值 0.00633) — agent 几乎不做有效位移
- `stability`: ep_sum=-1.416 (每步均值 0.00142) — 二次指数门控，量级极弱
- 终止：0/20 成功着陆，20/20 truncated@1000步

**Checkpoint 趋势（关键矛盾）：**
- cp70% score=38.6, landing_completion ep_sum=**91.9** (active 6.6%)
- cp80% score=**164.0**, landing_completion ep_sum=**273.1** (active 18.1%)
- cp100% score=106.7, landing_completion ep_sum=43.4 (active 3.1%)
- **最终评估**：landing_completion=0.000 (active 0.0%)

→ 训练过程中 agent **曾经学会着陆**（cp70-80% 时 landing_completion 激活），但最终策略**丢失了着陆能力**。说明稀疏 contact-gated bonus 造成奖励面 discontinuity：agent 偶然发现着陆获得巨大奖励 spike → policy 剧烈更新 → 失去着陆行为 → 退回"悬停 1000 步"的安全局部最优。

### behavior_diagnosis

Agent 当前行为：**全程悬停，不做有效位移，永不尝试着陆**。ep_sum 分布显示：
- `distance_progress` 每步仅 0.006（×5.0 缩放后），说明 agent 几乎停在原地
- `fuel_penalty` 占 magnitude_share 53.2%，每步稳定扣除
- 整个 episode 是"以极小代价悬停 1000 步"的懒惰策略

根因：**没有从高空到地面的连续梯度**。`distance_progress` 是等方向性的欧氏距离减少（水平漂移和垂直下降同等奖励），`landing_completion` 是稀疏的二值门控。Agent 卡在两个信号之间：距离奖励太弱无法驱动下降，着陆奖励太稀疏无法被发现。

### signal_completeness

对照环境卡片的 `reward_role_decomposition`：
- `approach_target` ✅ — distance_progress 覆盖，但等方向性导致垂直下降无特殊引导
- `stable_landing` ❌ — landing_completion 死组件，无有效着陆信号
- `action_penalty` ✅ — fuel_penalty 覆盖，但过于均匀（所有非零动作同等惩罚）
- **缺失信号**：从"接近目标区域"到"实际接触地面"之间的**垂直下降引导**。当前没有组件告诉 agent「当你已经在目标上方时，往下走是好的」。

### selected_level

**L2** — 将 landing 相关组件从"稀疏 contact-gated bonus"改为"连续 delta-y 下降奖励 + x-position 门控"。这是新的数学形态，不同于已穷尽的 proximity-gated 和 contact-gated 两种范式。

### selected_intervention

**替换 `landing_completion` 为 `descent_bonus`**：
- 组件名：`descent_bonus`
- 公式：当 `|x_pos| < 0.5`（agent 在目标垫水平范围内）时，奖励 `max(0, y_pos - ny_pos)`（垂直下降量）× 权重 × 门控系数
- 权重：3.0，与 distance_progress 的 ×5.0 量级匹配
- 门控：`max(0, 1 - |x_pos| / 0.5)`，越靠近中心奖励越强

**为什么这不是 proximity-gated（不会重复 iter 2 的 exploit）：**
- iter 2 的 proximity-gated landing 奖励的是**状态值**（在目标附近就给奖励 → 悬停收集）
- 本方案的 descent_bonus 奖励的是**状态变化量**（delta-y → 悬停时 reward=0）
- Agent 必须持续下降才能持续获得奖励，无法通过悬停 exploit

**同时保留 landing_completion 作为小额的最终成功标记**（降权到 0.5，避免 reward spike 导致策略崩溃），为 agent 提供"接触地面 = 好的"的最终确认信号。

### falsifiable_hypothesis

如果 descent_bonus 提供从高空到地面的连续梯度，agent 应该学会：（1）先水平移动到 x≈0（靠 distance_progress），（2）在 x≈0 时垂直下降到 y≈0（靠 descent_bonus），（3）接触地面完成着陆。预期结果：`descent_bonus` active_rate > 30%，ep_sum 显著为正，至少部分 episode 成功着陆（terminated > 0）。

### expected_next_round

- `descent_bonus`：active_rate 应 > 30%（agent 在目标上方时有下降行为），ep_sum_mean 应为正值
- `landing_completion`：可能仍然很低（因为它是最终接触信号），但不再是唯一着陆相关信号
- `distance_progress`：应略有增加（下降行为贡献距离减少）
- `fuel_penalty`：active_rate 可能略降（agent 在下降过程中可能使用 no_engine 滑行）
- `terminated`：期望从 0/20 变为 ≥ 1/20
- `score`：期望改善，但关键是行为变化而非绝对分数

### main_risk

- descent_bonus 可能引导 agent 高速坠落（垂直加速下降获取奖励），在接触地面时 crash。风险缓解：`stability` 组件的二次速度惩罚 + exp(-2d) 门控在近地时增强，应能抑制高速下降。如果下一轮出现大量 crash，需要增大 stability 权重或收紧门控。

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
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)          # 1 at x=0, 0 at |x|>=0.5
    descent = max(0.0, y_pos - ny_pos)                  # positive when descending
    descent_bonus = 3.0 * x_gate * descent

    # =====================================================================
    # D. landing_completion: small soft success bonus – both legs on ground,
    #    near centre, with low velocity and stable attitude.
    #    Reduced weight (0.5 instead of 2.0) to avoid reward-spike instability.
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
