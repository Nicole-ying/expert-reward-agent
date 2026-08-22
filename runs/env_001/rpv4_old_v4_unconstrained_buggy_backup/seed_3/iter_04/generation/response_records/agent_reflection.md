# Response Record

Now I have a complete picture. Let me analyze and produce the diagnosis.

---

## 诊断报告

### Evidence

| 来源 | 关键数据 |
|---|---|
| iter_2 (best=72.90) | contact_bonus ep_sum=2980, magnitude_share=99.6%, 直接距离惩罚 goal_proximity=-7.37, 20/20 terminated, len=253 |
| iter_3 (score=-50.45) | contact_bonus ep_sum=35, active_rate=0.0%, goal_proximity=1.016 (势能差), 12/20 terminated + 8/20 truncated, len=754 |
| iter_3 checkpoint | cp30% score=**100.4** (contact_bonus=5670) → cp100% score=**-98.7** (contact_bonus=35)：训练在 30% 处短暂找到着陆后**彻底崩溃** |
| iter_1 checkpoint | v1 势能差方案：cp10% contact_bonus=5.16 → cp20% 起 contact_bonus=0，**接触信号死亡** |
| 跨迭代对比 | 势能差方案（v1/v3）两次均失败（contact 死亡或训练崩溃）；直接距离惩罚（v2）成功学会着陆但 contact 主导 |

### Behavior Diagnosis

**iter_3 的行为模式是典型的训练崩溃**：

1. **势能差 goal_proximity 的致命缺陷**：v3 将 v2 成功的直接距离惩罚改为势能差（`delta_x_sq + delta_y_sq`），且权重从 v1 的 (1.0, 2.0) 大幅削弱到 (0.3, 0.6)。势能差信号的特点是：当 agent 静止不动时信号为零，只有位移才产生反馈。这导致 agent 在早期无法区分"停在远处"和"停在近处"——两者都不产生 goal_proximity 信号。相比之下，v2 的直接距离惩罚在任何位置都给出朝向目标的梯度。

2. **contact_bonus 的 per-step 累积漏洞**：v2 中 contact_bonus=100/step，agent 着陆后持续 ~30 步领取奖励，累计 2980（magnitude_share 99.6%）。这不仅淹没了所有其他信号，还在 iter_3 中引发灾难：cp30% 时 agent 短暂找到接触（contact_bonus=5670），但巨大的奖励尖峰破坏了价值函数，而微弱的势能差信号无法提供恢复所需的梯度，导致后续训练中着陆行为完全丢失。

3. **当前策略行为**：iter_3 最终策略在 20 集中有 8 集截断（未着陆），12 集虽然终止但 contact_bonus 仅 35（active_rate=0.0%），意味着 agent 几乎不再着陆，而是在空中徘徊直至超时或偶然终止。

### 干预方案（L2 — 改 contact_bonus 数学形态）

**核心语义修改**：将 contact_bonus 从 **per-step 状态奖励**（待在平台上每步 +100）改为 **一次性状态转移奖励**（首次接触 +200）+ **微小 per-step 停留激励**（+2/step）。

**同时回退 goal_proximity**：从势能差回退到 v2 验证过的直接距离惩罚，这是唯一稳定的方案。

**理由**：
- 一次性转移奖励杜绝"停留收集"漏洞——agent 无法通过赖在平台上累积天量奖励
- 直接距离惩罚提供持续、稳定的梯度，已被 v2 验证有效
- 微小停留激励（2/step ≈ v2 的 2%）确保 agent 着陆后有动机保持接触直到身体静止，但不会造成 dominance
- 这是单一语义修改（contact 从 per-step→transition），goal_proximity 回退是为配合该修改的必要联动

---

## 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: one-time landing bonus + direct distance gradient

    Key changes from v3:
    1. goal_proximity: revert from potential-based (delta squared distance)
       back to direct distance penalty — the only approach proven stable (v2).
       Provides consistent gradient toward (0,0) at all positions.
    2. contact_bonus: switch from per-step accumulation (100/step, which
       caused 99.6% magnitude dominance in v2) to a ONE-TIME transition bonus
       (+200 on first pad contact) plus a tiny per-step settling incentive (+2).
       This eliminates the "stay and collect" exploit while retaining the
       landing anchor.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur = obs[6]
    right_cont_cur = obs[7]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: direct distance penalty (v2 approach, proven stable) ---
    # Consistent gradient toward (0,0) at every step, regardless of movement.
    # Weight 0.05 means at typical start distance ~8, penalty ≈ -0.4/step;
    # over ~200-step approach accumulates ~-80, clearly beaten by +200 contact.
    w_dist = 0.05
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = -w_dist * dist_nxt

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Stronger damping near ground to prevent high-speed crashes.
    w_v = 0.15
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: ONE-TIME transition bonus + tiny settling incentive ---
    # The core fix: instead of 100/step (which dominated everything in v2
    # and caused training collapse in v3), reward the landing EVENT once,
    # then provide a gentle per-step incentive to stay until body settles.
    #
    # Detect whether agent was already on the pad last step (prevents repeat).
    near_target_cur = (abs(x_cur) < 0.5) and (abs(y_cur) < 0.3)
    both_legs_cur = (left_cont_cur > 0.5) and (right_cont_cur > 0.5)
    was_on_pad = near_target_cur and both_legs_cur

    near_target_nxt = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs_nxt = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    is_on_pad = near_target_nxt and both_legs_nxt

    # Transition: first step entering the pad region with both legs down.
    just_landed = is_on_pad and not was_on_pad

    w_contact_one_time = 200.0   # one-time reward for the landing event
    w_settle_per_step = 2.0      # tiny per-step incentive to maintain contact

    contact_bonus = 0.0
    if just_landed:
        contact_bonus += w_contact_one_time
    if is_on_pad:
        contact_bonus += w_settle_per_step

    # --- assemble ---
    total_reward = goal_prox + vel_penalty + orient_penalty + contact_bonus

    components = {
        'goal_proximity': goal_prox,
        'velocity_penalty': vel_penalty,
        'orientation_penalty': orient_penalty,
        'contact_bonus': contact_bonus
    }

    return total_reward, components
```
