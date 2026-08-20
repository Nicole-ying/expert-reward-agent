# Response Record

Now I have a complete picture. Let me compile the diagnosis.

---

## 1. 诊断报告

### evidence

**全局轨迹（iter 6→8，当前骨架）：**

| iter | score | len | term | landing_completion | descent_bonus | 关键变化 |
|------|-------|-----|------|--------------------|---------------|---------|
| 6 | -13.57 | 1000 | 0/20 | dead (0.00, 0%) | — | 新骨架，全截断 |
| 7 | 146.32 | 673 | 14/20 | 47.04, 19.0% | 3.28, 62.8% | 新增 descent_bonus (×3.0) |
| 8 | **199.42** | 478 | **17/20** | 70.98, 32.3% | 6.38, 69.7% | descent_bonus ×3.0→×6.0 |

**iter 8 详细数据：**
- 得分 199.42，距目标 200 仅差 **0.58**。range=[-41.00, 281.71]
- 终止：17/20 成功着陆，**3/20 truncated**（时间耗尽，1000步未完成）
- landing_completion magnitude_share=79.7%，接近但未超 80% 阈值
- stability active_rate=100%，全程负贡献（-1.87 ep_sum，magnitude仅2.1%，非 exploit）
- Checkpoint：cp100%=228.0（训练分布），final eval=199.4（固定种子），存在种子方差

**iter 7→8 的修改回顾：** 上一轮我将 descent_bonus 系数从 3.0 翻倍到 6.0。结果：得分 +53.1，len -195，term +3。修改方向正确且有效。

### behavior_diagnosis

策略已经学会了基本着陆：17/20 集成功终止，平均 478 步完成。但 **3 集仍然截断**，这些 episode 耗尽 1000 步未能触发终止条件，score range 下限 -41 说明这些截断 episode 拉低了平均分。这 3 集很可能：agent 在水平方向偏移较大（|x|>0.5），descent_bonus 的 x_gate 将其完全屏蔽，仅靠 distance_progress（各向同性）引导，缺乏「靠近中心 + 同时下降」的联合梯度，导致徘徊或过慢下降。

### signal_completeness

| 职责 | 覆盖情况 | 评估 |
|------|---------|------|
| approach_target（接近） | distance_progress + descent_bonus | descent_bonus 仅在 \|x\|<0.5 激活，水平偏移较大时缺失垂直引导 |
| stable_landing（着陆） | landing_completion | 正常工作，32.3% active，79.7% share，已是主导信号 |
| action_penalty（节能） | fuel_penalty | 正常工作 |
| stability（稳定） | stability（exp门控惩罚） | 100% active 但 magnitude 仅 2.1%，非主要问题 |

**缺口：** 当 agent 水平偏移 0.5 < |x| < ~1.0 时，descent_bonus 完全沉默。agent 只能依赖 distance_progress（欧氏距离减少 = 5.0），它对水平和垂直移动的区分度不足。需要一个更宽的「居中 + 下降」联合信号。

### selected_level

**L1** — 仅调整参数，不改变数学形态。当前信号框架已被验证有效（iter 7→8 大幅改善），仅需针对截断 episode 微调 descent_bonus 的 x_gate 宽度。

### selected_intervention

**目标组件：`descent_bonus`**  
**修改：x_gate 宽度参数从 0.5 → 0.8**

```python
# 当前（iter 8）:
x_gate = max(0.0, 1.0 - abs(x_pos) / 0.5)   # 仅在 |x|<0.5 激活

# 修改后（iter 9）:
x_gate = max(0.0, 1.0 - abs(x_pos) / 0.8)   # 在 |x|<0.8 即开始激活
```

效果：当 agent 在 0.5<|x|<0.8 范围内，将获得部分下降激励（线性衰减：|x|=0.5 时 gate=0.375，|x|=0.7 时 gate=0.125），鼓励「边下降边居中」而非「先完全居中再下降」。

### falsifiable_hypothesis

截断的 3 集 agent 在水平偏移 0.5~1.0 范围内徘徊，因为 descent_bonus 不激活，它们缺少明确的「下降」梯度。加宽 x_gate 后，这些 agent 将在偏移较大时就开始获得下降激励，应能更快接近 pad 并触发 landing_completion → termination。预期 truncated 从 3/20 降至 ≤1/20，得分突破 200。

### expected_next_round

| 指标 | 当前 (iter 8) | 预期 (iter 9) | 理由 |
|------|-------------|-------------|------|
| score | 199.42 | ≥200 | 截断 episode 减少，平均得分上升 |
| len | 477.65 | ≤450 | 更快完成 |
| terminated | 17/20 | ≥18/20 | 更多 episode 成功着陆 |
| descent_bonus ep_sum | 6.38 | 7~8 | 更宽门控 → 更多步获得下降奖励 |
| descent_bonus active_rate | 69.7% | 75~82% | 更多步满足门控条件 |
| landing_completion ep_sum | 70.98 | 65~75 | 可能略降（门放宽后 agent 可能以稍低质量着陆），但总体仍保持 |
| landing_completion magnitude_share | 79.7% | 70~78% | 其他组件（descent_bonus）增长，相对份额可能略降 |

### main_risk

**过早下降导致 crash**：如果 agent 在 |x|≈0.7 时收到下降激励并积极响应，可能在未充分居中时下降，导致落在 pad 边缘或外侧 → crash。但线性 gate 在 |x|=0.7 时仅 0.125，激励较弱，且 distance_progress 同时提供居中信号，应能平衡。如果下一轮 crash 率上升，回退此参数即可。

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
    #    v9 change: x_gate widened from /0.5 to /0.8 to give partial descent
    #    incentive to agents moderately offset horizontally (0.5<|x|<0.8),
    #    helping them descend while centering instead of requiring perfect
    #    alignment first.  This targets the 3/20 truncated episodes from v8.
    # =====================================================================
    x_gate = max(0.0, 1.0 - abs(x_pos) / 0.8)          # 1 at x=0, 0 at |x|>=0.8
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
