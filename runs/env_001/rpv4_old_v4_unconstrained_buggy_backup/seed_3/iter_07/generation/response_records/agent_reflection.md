# Response Record

Now I have a comprehensive understanding. Let me compile the full diagnosis.

---

## 诊断报告

### Evidence

**历史轨迹：**

| iter | skeleton | score | contact active | contact ep_sum | 关键变化 |
|---:|---|---:|---:|---:|---|
| 1 | dist_penalty + contact(100) | -32 | 0% | 0 | 从未着陆 |
| 2 | dist_penalty + contact(100) | 73 | 11.8% | 2980 | **首次成功着陆**，但 contact 占 99.6% |
| 3 | potential(sq) + contact(100) | -50 | 0% | 35 | 换势能公式，接触死亡 |
| 4 | one-time transition | -109 | 1.8% | 142 | 一刀切奖励，agent 快速坠毁 |
| 5 | dist_penalty + contact(25) | -96 | 0% | 0 | contact 权重过低，接触死亡 |
| **6** | **potential(euclidean) + contact(80)** | **154** | **13.7%** | **4548** | **历史最佳，但 contact 占 99.7%** |

**iter_6 checkpoint 趋势的关键发现：**

| 阶段 | score | contact_bonus | goal_proximity | velocity_penalty |
|---|---|---|---|---|
| cp80% | **240.2** ✅ | 3888 | 2.703 | -8.401 |
| cp90% | 197.6 | 4316 | 2.383 | -6.675 |
| cp100% | 160.2 | 4540 | 2.242 | -6.194 |

- **Score 在 cp80% 已达 240（超过目标 200），但随后持续下降到 160。**
- 同期 contact_bonus 从 3888 上升到 4540（+17%），而 goal_proximity 从 2.703 下降到 2.242（-17%），velocity_penalty 从 -8.4 减弱到 -6.2（速度控制变差）。
- 这是典型的 **reward exploitation → 策略退化**：agent 逐步学会最大化 contact_bonus（在平台上待更久），但同时放弃了高效的接近策略。

**当前 iter_6 最终评估：**
- score=154，距目标 200 差 23%
- contact_bonus magnitude_share=99.7%，其他三个组件合计仅 0.3%
- len=415（偏长），terminated=20/20（全部着陆成功）

### Behavior Diagnosis

agent 已经学会稳定着陆（20/20 terminated），但存在两个问题：

1. **contact_bonus 压倒性主导（99.7%）**：agent 几乎完全被 per-step contact bonus (80/step) 驱动。一旦学会着陆，其他信号（高效接近、速度控制、姿态稳定）都被淹没在噪声中。

2. **训练中后期策略退化**：cp80% 时 agent 有更高效的接近策略（goal_proximity 更高、速度更快），score 达到 240。但随着训练继续，agent 发现"在平台上多待一会儿"可以轻松赚取更多 contact_bonus，于是放弃了高效接近，导致 score 从 240 退到 160。

**根因**：per-step contact_bonus 权重过高（80），导致 agent 在学会着陆后没有动力优化接近效率。goal_proximity（w_pot=2.0）的信号太弱，无法在训练后期与 contact_bonus 竞争。

### 干预方案

**L1 — 系数调整**（距目标 23%，优先 L1）：

1. **w_contact: 80 → 40**：将 contact_bonus 减半，降低其压倒性主导地位，让其他信号有呼吸空间。仍保留足够的着陆激励（iter_2 用 100 成功，iter_5 用 25 失败，40 是安全中间值）。

2. **w_pot: 2.0 → 6.0**：将 goal_proximity 的势能差权重提高 3 倍。在 cp80% 行为水平下，goal_proximity 将从 ~2.7 提升到 ~8.1，使接近效率的信号在训练后期仍可与 contact_bonus 竞争。

3. **w_v: 0.3 → 0.5**：适度加强近地速度惩罚，防止 agent 因 goal_proximity 增强而采取过激的高速俯冲。

4. **w_ang: 0.05 → 0.08**：微调姿态惩罚，提高着陆姿态质量。

**预期效果**：contact_bonus 仍是最强信号但不再压倒一切（预计占比从 99.7% 降至 ~95-97%），goal_proximity 足够强以维持接近效率，避免中后期退化。

---

## 完整代码

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: rebalanced coefficients — moderate contact + strong approach gradient

    Diagnosis from v6 (score=154, best so far but 23% below target):
    - contact_bonus dominated at 99.7% magnitude share, drowning all other signals.
    - Checkpoint trend revealed score peaked at cp80% (240, above target) then
      degraded to cp100% (160). At cp80%, contact_bonus was lower (3888) and
      goal_proximity was higher (2.703) — the agent had efficient approach.
    - As training progressed, agent learned to maximize contact_bonus by spending
      more time on pad, while approach efficiency decayed (goal_proximity fell).
    - Root cause: w_pot=2.0 too weak to compete with w_contact=80 during late training.

    Changes from v6:
    1. w_contact: 80 → 40 — halve per-step contact bonus to reduce dominance.
       Still well above iter_5's failed w=25, but low enough to let other signals
       shape behavior during late training.
    2. w_pot: 2.0 → 6.0 — 3x stronger potential-based progress reward.
       At cp80% behavior levels, goal_proximity would be ~8.1 instead of ~2.7,
       making approach efficiency a meaningful signal throughout training.
    3. w_v: 0.3 → 0.5 — stronger velocity damping near ground to prevent
       aggressive dives that a stronger w_pot might otherwise encourage.
    4. w_ang: 0.05 → 0.08 — slightly stronger orientation penalty for
       better landing posture quality.

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

    # --- 1. goal_proximity: potential-based progress reward ---
    # Rewards reduction in Euclidean distance to target (0,0).
    # Positive when approaching, negative when retreating.
    # Weight 6.0 (up from 2.0): provides strong approach gradient that
    # remains competitive with contact_bonus throughout training,
    # preventing the late-training degradation seen in v6.
    w_pot = 6.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.5 (up from 0.3) provides stronger
    # braking incentive near the pad to ensure soft landings.
    w_v = 0.5
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Slightly increased angle weight (0.05→0.08) for better landing posture.
    w_ang = 0.08
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (iter_2/v6 proven): rewards every step with both legs
    # on pad near target. Weight 40 (down from 80): still the largest single
    # signal but no longer at 99.7% dominance. This lets goal_proximity and
    # velocity_penalty meaningfully shape approach behavior.
    w_contact = 40.0
    near_target = (abs(x_nxt) < 0.5) and (abs(y_nxt) < 0.3)
    both_legs = (left_cont_nxt > 0.5) and (right_cont_nxt > 0.5)
    contact_bonus = w_contact if (near_target and both_legs) else 0.0

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
