# Response Record

Now I have a complete picture. Let me synthesize the diagnosis.

---

## 1. 诊断报告

### evidence

**核心模式 — contact_bonus 的「发现-遗忘」循环：**

| iter | contact_bonus ep_sum | active_rate | score | terminated |
|------|---------------------|-------------|-------|------------|
| 1 | 0.000 | 0.0% | -32.35 | 0/20 |
| 2 | 154.465 | 2.8% | 107.13 | 11/20 |
| 3 | 0.000 | 0.0% | -26.17 | 0/20 |
| 4 | 114.083 | 4.9% | 183.52 | 19/20 |
| 5 | 0.000 | 0.0% | -15.01 | 0/20 |
| 6 | 0.000 | 0.0% | -15.73 | 0/20 |

6 轮中 contact_bonus 仅 2 轮激活（iter_2, iter_4），激活时分数达标，否则失败。这是典型的「彩票」— agent 偶然发现着陆行为获得高分，但下一轮随机种子改变后丢失。

**Checkpoint 证据 — iter_6 训练中期发现了着陆但最终遗忘：**

| checkpoint | score | contact_bonus ep_sum |
|------------|-------|---------------------|
| cp10%-cp70% | -61 ~ -6.5 | 0.000（死） |
| cp80% | **156.2** | **1168.992**（骤升，发现着陆） |
| cp90% | 89.4 | 136.610（正在丢失） |
| cp100% | -15.0 | 0.000（完全遗忘） |

agent 在 cp80% 学会了着陆（contact_bonus 暴增至 1169），但随后的训练中该行为崩溃。巨大的 contact_bonus 奖励脉冲（1169 vs 其他组件总和约 10）造成了梯度冲击，策略更新后着陆行为被覆盖。

**descent_incentive 演变 — 连续 2 轮强化但未转化为着陆：**

| iter | w_desc | gate | ep_sum | active_rate | 是否着陆 |
|------|--------|------|--------|-------------|---------|
| 4 | 2.0 | y<0.8 线性 | 0.779 | 72.4% | ✅ (靠 contact_bonus 彩票) |
| 5 | 2.0 | y<0.8 线性 | 0.624 | 66.1% | ❌ |
| 6 | 6.0 | y<1.5 线性 | 3.366 | 73.8% | ❌ |

iter_6 的 descent_incentive ep_sum 比 iter_4 增长了 4.3 倍（0.78→3.37），但着陆率却从 19/20 降为 0/20。**这说明 descent_incentive 的量级提升并没有转化为着陆能力**——它让 agent 更多地在近地区域下降，但无法推动 agent 完成最后几厘米的 touchdown。

**descent_incentive 为什么在最关键的最后阶段失效：**

当前公式：`w_desc * near_ground * near_center * descending`，其中 `descending = max(0, y_cur - y_nxt)`。

问题在于物理约束：agent 越接近地面（y→0），单步能下降的幅度越小（重力加速有限，且需要减速避免撞击）。`descending` 在 y→0 时自然萎缩，而线性 gate `(1-y/1.5)` 的增长不足以补偿。结果是：**descent_incentive 在最需要强信号的最后几厘米反而最弱**。

例如 y=0.3→0.25 时 gate=0.8，descending=0.05，reward=6×0.8×0.05=0.24；而 y=0.05→0.03 时 gate=0.967，descending=0.02，reward=6×0.967×0.02=0.116。信号减半。

### behavior_diagnosis

agent 当前行为：在目标上方悬停/缓慢下降（descent_incentive active_rate 73.8%），收集小幅正奖励（goal_proximity + descent_incentive ≈ 11.2），被速度和姿态惩罚抵消后净奖励接近零。agent 学会了「靠近目标并缓慢移动」但没有动力完成最后的 touchdown——因为 descent_incentive 在 y<0.3 后梯度不增反减。contact_bonus 是唯一能推动 touchdown 的信号，但它是稀疏的（需要双腿同时接触，只在 y≈0 且姿态正确时才触发），agent 无法从零开始稳定学会。

### signal_completeness

| 职责 | 状态 | 问题 |
|------|------|------|
| goal_proximity（接近目标） | ✅ 完备 | delta squared-distance，6 轮稳定 |
| velocity_damping（减速） | ✅ 完备 | 指数高度门控，合理 |
| orientation_penalty（姿态） | ✅ 完备 | 角度+角速度惩罚 |
| **descent→touchdown 梯度** | ❌ 薄弱 | 线性 gate × 下降率，在 y→0 时梯度萎缩 |
| contact confirmation | ⚠️ 稀疏/不稳定 | 双腿接触二值门控导致彩票效应 |

缺失的关键信号：**在 y→0 时强度递增而非递减的 touchdown 梯度**。

### selected_level

**L2** — 修改 descent_incentive 的数学形态。L1（调权重）已在 iter_5 和 iter_6 尝试两次（w 2→5→6），均未解决根本问题。需要改变信号结构。

### selected_intervention

**目标组件：descent_incentive**。将当前的「线性高度门控 × 下降率」改为 **potential-based shaping（势能差奖励）**：

- 势函数：`potential(y) = 1 / (y + 0.1)`，y 越小势能越大
- 奖励 = `w × near_center × max(0, potential(y_next) - potential(y_cur))`
- 权重 w=1.5（势能差本身已提供足够量级）

**为什么这解决了梯度萎缩问题：**

势能差 `Δpotential = 1/(y_next+0.1) - 1/(y_cur+0.1)`。当 agent 以恒定速度下降时，在 y 较小时 Δpotential 更大：
- y 0.30→0.25：1/0.35 - 1/0.40 = 2.857 - 2.500 = 0.357
- y 0.10→0.08：1/0.18 - 1/0.20 = 5.556 - 5.000 = 0.556
- y 0.04→0.03：1/0.13 - 1/0.14 = 7.692 - 7.143 = 0.549

即使 descending 缩小，势能差仍保持强劲甚至增大。这创建了一个真正的 funnel：越接近 touchdown，梯度越强。

**联动修改：contact_bonus 权重 4.0→2.0**，服务于同一语义「创建平衡的连续 landing 梯度」：防止 contact_bonus 在 agent 学会着陆后再次因奖励脉冲（1168→0 骤降）导致策略崩溃。

### falsifiable_hypothesis

势能差公式使 descent_incentive 在 y<0.3 区域的 per-step 梯度比当前线性 gate 强 2-3 倍，且随 y→0 递增而非递减。这应该能让 agent 在不需要 contact_bonus 运气的情况下，通过连续梯度学会完成 touchdown。

### expected_next_round

- descent_incentive ep_sum：预期从 3.37 上升到 8-12（势能差累积更强）
- descent_incentive active_rate：预期保持 70-80%（near_center 门控不变）
- 最关键：terminated 应 > 0（至少出现 1-2 次成功着陆），contact_bonus active_rate > 0%
- checkpoint 趋势：期望 cp80%→cp100% 期间 contact_bonus 不再归零（着陆行为稳定保持）
- 分数：预期从 -15 改善到正值区间（0-60），如果着陆成功可接近 150+

### main_risk

势能差公式在 y 较小时的强梯度可能导致 agent 过度追求快速下降而在触地前未能充分减速——即从「不敢着陆」变成「硬着陆」。velocity_penalty（指数高度门控）应在近地处提供制衡。如果下一轮出现高速撞击，需要调高 velocity_penalty 的 w_v 或收紧 h_scale。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7 reward: L2 — potential-based descent_incentive replaces linear-gate formulation.

    Problem (v6): descent_incentive uses `near_ground * descending` where both terms
    are linear. As y→0, the possible descent per step shrinks (physics), causing the
    gradient to weaken at the most critical moment — the final few cm before touchdown.
    The agent hovers near the pad but cannot bridge the last gap to landing.

    Change:
      - descent_incentive: replaced linear `(1-y/1.5) * (y_cur-y_nxt)` with
        potential-based shaping. Potential φ(y) = 1/(y+0.1). Reward = φ(y_nxt) - φ(y_cur).
        This creates a gradient that naturally INTENSIFIES as y→0, compensating for
        the shrinking descent step size.
      - contact_bonus: weight reduced 4.0→2.0 (supporting change). When contact_bonus
        fires at cp80% (1169 ep_sum), it creates a destabilizing reward spike that
        causes catastrophic forgetting of landing behavior. Lower weight reduces the
        spike while keeping it as a meaningful confirmation signal.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity (improvement_delta, squared distance) ---
    w_x, w_y = 2.0, 4.0
    delta_x_sq = x_cur ** 2 - x_nxt ** 2
    delta_y_sq = y_cur ** 2 - y_nxt ** 2
    goal_prox = w_x * delta_x_sq + w_y * delta_y_sq

    # --- 2. velocity_damping (exponential height-gated) ---
    w_v = 0.5
    h_scale = 1.0
    gate = 2.718281828 ** (-y_nxt / h_scale)
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty (absolute angle + squared angular velocity) ---
    w_ang = 0.1
    w_angvel = 0.01
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. descent_incentive (L2 — potential-based shaping) ---
    # Potential φ(y) = 1/(y+eps): increases as y decreases, with derivative
    # dφ/dy = -1/(y+eps)^2 that grows strongly near y=0. This creates a
    # natural funnel: the same descent step yields MORE reward when closer
    # to the ground, exactly where the linear formulation used to weaken.
    #
    # Gated by near_center so agent only gets descent credit when aligned
    # with the landing pad.
    w_desc = 1.5
    eps = 0.1
    potential_cur = 1.0 / (y_cur + eps)
    potential_nxt = 1.0 / (y_nxt + eps)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)  # active within |x|<0.5
    descent_incentive = w_desc * near_center * max(0.0, potential_nxt - potential_cur)

    # --- 5. contact_bonus (sparse landing confirmation, reduced weight) ---
    # Weight reduced 4.0→2.0 to prevent the reward spike (observed up to 1169
    # ep_sum at cp80%) from destabilizing training and causing catastrophic
    # forgetting of landing behavior.
    w_contact = 2.0
    y_th = 0.1
    x_th = 0.2
    y_close = max(0.0, 1.0 - abs(y_nxt) / y_th)
    x_close = max(0.0, 1.0 - abs(x_nxt) / x_th)
    both_legs = left_cont_nxt * right_cont_nxt
    contact_bonus = w_contact * y_close * x_close * both_legs

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + descent_incentive + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'descent_incentive': descent_incentive,
        'contact_bonus': contact_bonus
    }

    return (float(total_reward), components)
```
