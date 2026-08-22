# Response Record

Now I have a full picture. Let me compile the diagnosis.

---

## 诊断报告

### Evidence

**历史演进（contact_bonus 为核心线索）：**

| iter | contact 机制 | contact ep_sum | mag_share | score | len |
|---:|---|---:|---:|---:|---:|
| 6 | per-step 80/step | 4548 | 99.7% | 154 | 415 |
| 7 | per-step 40/step | 32 | 64% | -121 | 772 |
| 8 | per-step 65/step | 3695 | 99.7% | 154 | 415 |
| **9** | **transition 500+3/step** | **1031** | **99.2%** | **219** | **474** |

**v9 关键变化：** per-step → transition-based 将 contact ep_sum 从 3695 压缩到 1031（-72%），score 从 154 跃升至 219（+42%）。这是正确的方向。

**v9 当前状态：**
- score=219.65，已超过目标 200（+9.8%）
- terminated=19/20，truncated=1/20 — 大部分自然终止
- contact_bonus 仍占 99.2% magnitude_share
- sustain 贡献估算：ep_sum=1031，landing=500 → sustain ≈ 531，约 177 步 sustain
- sustain（531）与 landing（500）几乎平起平坐 — **agent 被激励「停留」几乎和「着陆」一样多**

**训练不稳定性（checkpoint 趋势）：**
- score：59.8 → -150.9 → -175.6 → -42.4 → **-228.5** → 131.3 → 139.8 → **226.8** → 124.1 → 223.7
- contact_bonus 在 cp70% 暴增至 2038，cp80% 回落至 736：典型的 exploit→correction 循环
- 训练中期大幅振荡说明 reward 中仍有可被 exploit 的持续性激励

### Behavior Diagnosis

当前策略已能成功着陆（19/20 terminated），但 **sustain_bonus=3.0/step 仍创造了有意义的 over-stay 激励**。landing_bonus=500 与 sustain≈531 平分秋色，agent 被同等激励去「着陆」和「赖着不走」。这导致：
1. 训练过程中 agent 周期性 exploit sustain（cp70% 时 contact 飙到 2038）
2. 其他效率组件（goal_proximity、velocity_penalty）被完全淹没（合计 <1%）
3. 虽然最终 score 达标，但训练路径不稳定

### 干预方案：L1 — 降低 sustain_bonus

**选择 L1 的理由：** score 已超过目标 200，差距 <10%（实际是 +9.8%）。历史教训（v7 contact 40→collapse）反复证明：score 接近目标时激进改结构（L2）容易崩盘。v9 的 transition-based 框架已证明正确，只需微调系数。

**具体修改：** `w_sustain`: 3.0 → 1.0

- 预期效果：sustain 从 ~531 降至 ~177，landing（500）将成为 contact 的主导信号（~74% vs ~26%），减少 over-stay 激励
- 风险低：1.0 仍 > 0，保留「留在平台上」的弱梯度，不会像 v7 那样导致 landing 行为崩溃
- 这是对 v9 方向的延续：v9 将 per-step 从 65→3（22x 降），现在从 3→1（3x 降），进一步纯化 landing 信号

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v10: L1 — reduce sustain_bonus 3.0→1.0 to further curb over-stay

    Diagnosis from v9 (score=219.65, above target 200):
    - Transition-based contact (500 landing + 3 sustain) successfully boosted
      score from 154→219 by fixing per-step over-stay exploitation.
    - However, sustain_bonus at 3.0/step still accumulates ~531 points on average
      (177 sustain steps), nearly equal to the landing_bonus (500).
    - Training remains volatile: contact spikes to 2038 at cp70%, score oscillates
      between -228 and +226 during training — classic exploit→correction cycles.
    - Other efficiency components (goal_proximity, velocity, orientation) remain
      at <1% magnitude share, drowned out by contact.

    Change from v9 (L1: coefficient only):
    1. w_sustain: 3.0 → 1.0 — 3x reduction in per-step stay incentive.
       Landing (500) becomes ~74% of contact signal, sustain ~26%.
       Retains weak gradient to stay on pad until natural termination,
       but no longer competes with landing as an equally rewarding strategy.

    All other components unchanged from v9 proven backbone.

    obs:      [x, y, vx, vy, angle, ang_vel, left_cont, right_cont]
    next_obs: same layout after taking action
    """
    # --- unpack observations (current + next) ---
    x_cur, y_cur = obs[0], obs[1]
    left_cont_cur, right_cont_cur = obs[6], obs[7]

    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Weight 2.0 (v6/v8/v9 proven): consistent approach gradient.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6/v8/v9 proven).
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6/v8/v9 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: transition-based landing reward ---
    # v9 design (proven): one-time landing bonus + tiny sustain.
    # v10 change: sustain 3.0 → 1.0 (L1) to further reduce over-stay.
    #
    # landing_bonus (500, one-time): rewards the FIRST step entering safe contact.
    #   Concentrated signal strong enough to drive landing learning.
    # sustain_bonus (1.0/step, was 3.0): weak gradient to stay on pad until
    #   natural termination. At 1.0, even 200 stay-steps = only 200 points
    #   vs 500 for landing — landing is clearly the dominant incentive.
    #
    # Safe contact: both legs on pad, near target center.
    w_land = 500.0      # one-time landing bonus
    w_sustain = 1.0     # per-step sustain (v9: 3.0 → v10: 1.0, 3x reduction)

    near_target_cur = (abs(x_cur) < 0.5) and (abs(y_cur) < 0.3)
    both_legs_cur = (left_cont_cur > 0.5) and (right_cont_cur > 0.5)
    was_safe = near_target_cur and both_legs_cur

    near_target_nxt = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs_nxt = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    is_safe = near_target_nxt and both_legs_nxt

    # Transition: first step entering safe contact
    landing_bonus = w_land if (is_safe and not was_safe) else 0.0
    # Sustain: every step maintaining safe contact
    sustain_bonus = w_sustain if is_safe else 0.0

    contact_bonus = landing_bonus + sustain_bonus

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
