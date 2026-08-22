# Response Record

---

## 1. 诊断报告

### evidence

| 来源 | 关键数据 |
|---|---|
| iter_4 feedback | score=183.52, target=200, gap=**16.48 (8.2%)**, terminated=19/20, truncated=1/20 |
| iter_4 components | contact_bonus=114.08 (86.3% mag), goal_proximity=7.95 (6.2%), velocity_penalty=-5.74, orientation_penalty=-3.39, descent_incentive=0.78 (0.6%) |
| checkpoint trend | cp90%=2.3 → cp100%=187.5，contact_bonus 在 cp90% 首次激活(54.7)→cp100%(104.4)，着陆行为在训练末期集中涌现 |
| iter_2 (best predecessor) | w_contact=10.0 → score=107, contact_bonus=154.5, mag_share=92.5% |
| iter_3 (crash) | w_contact 降至 4.0 且无 descent_incentive → score=-26, contact_bonus 死亡 |
| iter_4 (current best) | w_contact=4.0 + descent_incentive → score=183.5, contact_bonus=114.1, 着陆 19/20 |
| descent_incentive 历史 | 仅 iter_4 存在：ep_sum=0.78, active_rate=72.4%，贡献极微弱 — 其真正作用是提供可微分梯度让 agent 发现着陆区，而非数值贡献 |

### behavior_diagnosis

agent 已学会在 19/20 场景中成功着陆，1 个 episode 被 truncate（超时）。策略行为基本正确，但总奖励距目标差 ~8%。核心问题不是行为缺陷，而是 **terminal reward 的量级偏保守**——contact_bonus 在 w=4.0 时产生 114 ep_sum，而在 iter_2 中 w=10.0 时曾产出 154.5。当前权重处于「刚好能让 descent_incentive 引导 agent 发现着陆」的下临界区，而不是「充分奖励正确行为」的舒适区。

### signal_completeness

五大职责全部覆盖且均存活：goal_proximity（接近引导）、velocity_penalty（软着陆）、orientation_penalty（姿态稳定）、descent_incentive（下降梯度）、contact_bonus（着陆确认）。无缺失信号。

### selected_level

**L1** — 差距 8.2%（<10%），架构已被 iter_4 证明有效，只需系数微调即可跨过 200 门槛。

### selected_intervention

**`contact_bonus` 权重：4.0 → 5.0**（+25%）。

这是唯一一个可以通过单一系数调整产生 ~28 分增量的组件。descent_incentive（0.78/ep）即使翻 5 倍也仅 +3.1，goal_proximity 翻 1.5 倍仅 +4.0，均无法独立弥合 16.5 的缺口。contact_bonus 的量级与权重近似线性（iter_2: w=10→154.5, iter_4: w=4→114.1），w=5.0 预计产出 ~142，为总分贡献 +28。

### falsifiable_hypothesis

contact_bonus 从 4.0→5.0 将通过等比放大已成功的着陆奖励，使平均得分从 183.5 提升至 ~205-215。由于 agent 已稳定着陆 19/20，这不是「买分」，而是对已习得的正确行为给予更充分的正向强化。velocity_penalty 和 orientation_penalty 的绝对值不变，着陆质量约束保持原有力道。

### expected_next_round

| 指标 | 当前 | 预期 |
|---|---|---|
| score | 183.5 | 200–215 |
| contact_bonus ep_sum | 114.1 | 135–150 |
| contact_bonus magnitude_share | 86.3% | 88–91% |
| terminated | 19/20 | 19–20/20 |
| 其他组件 | 基本不变 | ±5% 波动 |

### main_risk

contact_bonus magnitude_share 将从 86% 升至 ~90%，velocity/orientation 惩罚的相对梯度进一步被稀释。极端情况下 agent 可能容忍稍高的着陆速度。但两个惩罚的绝对量级不变，且当前 19/20 已成功着陆，实际劣化概率低。若下一轮出现速度或姿态劣化，回退 w_contact 并转 L2 加强 velocity_penalty 的门控即可。

---

## 2. 完整 Python 代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v5 reward: L1 — contact_bonus weight 4.0→5.0 to close the 8% gap to target 200.

    All other components unchanged from v4. The descent_incentive (added in v4)
    successfully enabled the agent to discover landing; now we amplify the
    terminal landing reward to push the mean score above 200.

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

    # --- 4. descent_incentive (continuous descent gradient near landing zone) ---
    w_desc = 2.0
    near_ground = max(0.0, 1.0 - y_nxt / 0.8)
    near_center = max(0.0, 1.0 - abs(x_nxt) / 0.5)
    descending = max(0.0, y_cur - y_nxt)
    descent_incentive = w_desc * near_ground * near_center * descending

    # --- 5. contact_bonus (sparse landing event bonus, weight ↑ 4.0→5.0) ---
    w_contact = 5.0
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
