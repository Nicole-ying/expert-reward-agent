# Response Record

## 诊断报告

### Evidence

**历史总览（8轮迭代）**：

| iter | w_contact | contact ep_sum | active_rate | score | best | 关键事件 |
|:---:|:---:|---:|---:|---:|---:|---|
| 1 | 100(sparse) | 0 | 0% | -32 | -32 | 从未着陆 |
| 2 | 100 | 2980 | 11.8% | 73 | 73 | 首次成功着陆 |
| 3 | ~squared | 35 | 0% | -50 | 73 | contact 崩溃 |
| 4 | — | 142 | 1.8% | -109 | 73 | 继续崩溃 |
| 5 | 25 | 0 | 0% | -96 | 73 | 完全死亡 |
| **6** | **80** | **4548** | **13.7%** | **154** | **154** | ✅ 新 best，但 contact 99.7% |
| 7 | 40 | 32 | 0.1% | -120 | 154 | contact 再次崩溃 |
| 8 | 65 | 3695 | 13.7% | 154 | 154 | 回到 v6 水平，无改善 |

**v8 checkpoint 趋势（本轮核心证据）**：

| 阶段 | score | contact ep_sum | goal_proximity | 解读 |
|---|---|---|---|---|
| cp40% | 70.9 | **7975.5** (峰值) | 2.30 | agent 发现大量刷 contact |
| cp80% | **240.2** (峰值) | 3159 (谷值) | **2.703** (峰值) | 效率最高：contact 少、接近快 |
| cp100% | 160.2 | 3688.75 (反弹) | 2.242 (回落) | contact 反弹，效率退化 |

**关键发现**：
- cp80% 时 score=240.2，**已超过目标 200**，证明架构本身有能力达标
- cp80%→cp100%：contact 从 3159 反弹到 3689（+17%），goal_proximity 从 2.703 跌到 2.242（-17%），score 从 240 跌到 160（-33%）
- 这是经典的 **per-step contact bonus 导致的过留（over-stay）退化**：agent 在学会高效着陆后，又逐渐滑向"着陆后赖着不走刷 contact"的局部最优

**v7 教训**：w_contact 从 80→40 直接导致 contact active_rate 从 13.7% 崩塌到 0.1%（score=-120）。单纯降权重不可行——contact 信号必须足够强才能驱动着陆学习，但 per-step 结构必然导致过留。

### Behavior Diagnosis

当前策略行为模式：agent 学会了着陆（terminated=20/20，len=415），但在着陆后持续停留收集 contact_bonus（ep_sum=3695，约 57 步接触）。contact_bonus 以 99.7% 的 magnitude_share 碾压所有效率信号（goal_proximity 仅 0.1%），导致 agent 没有动力快速完成着陆后收尾。训练后期 agent 在"高效着陆"和"赖着刷分"之间反复振荡，无法稳定在 cp80% 的高效状态。

**根因**：per-step contact bonus 将「持续停留在着陆状态」错认为「着陆成功」来奖励。agent 学会着陆后，每多待一步就多拿 65 分——这远大于 goal_proximity（~2.3/episode）和 velocity_penalty（~-6.6/episode）的量级。

### 干预方案

**L2 修改**：将 contact_bonus 从 per-step 持续奖励改为 **状态转移事件奖励**（"持续→状态转移事件"模式）：

1. **landing_bonus（一次性）**：在 agent **首次**实现安全接触（near_target + both_legs）时给予大额奖励。这是着陆事件的集中信号，确保 agent 有足够动力学习着陆。
2. **sustain_bonus（per-step，小额）**：维持接触后每步给予极小奖励，提供温和的"留在原地等待终止"梯度，但远不足以驱动过留。

这保持了着陆激励的强度（一次性 500 ≈ 旧方案 8 步），同时消除了过留的增量收益（旧方案每多待一步 +65，新方案仅 +3）。

**为什么不做 L1**：v7（w=40）、v8（w=65）均已证明纯权重调参无法解决 per-step 结构的过留问题。v8 的 w=65 已是"不崩溃的最低有效权重"，但 99.7% 的 magnitude_share 仍然导致退化。必须换结构。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: transition-based contact reward — fix per-step over-stay degradation

    Diagnosis from v8 (score=154, cp80% peaked at 240 then degraded to 160):
    - Per-step contact_bonus (w=65/step) created a persistent over-stay incentive.
      Agent learned to land then stayed collecting ~57 steps of bonus (3695 total),
      drowning out efficiency signals (goal_proximity ~2.3, velocity ~-6.6).
    - cp80%→cp100%: contact rebounded +17%, approach efficiency dropped -17%,
      score fell from 240→160. Classic per-step reward exploitation.
    - Weight tuning alone (v6=80, v7=40→collapsed, v8=65) cannot fix this:
      contact must be strong enough to drive landing, but per-step always over-stays.

    Changes from v8:
    1. contact_bonus: restructured from per-step accumulation to transition-based:
       - landing_bonus (500, one-time): rewards the FIRST step achieving safe contact.
         Equivalent to ~8 old steps — concentrated signal for learning to land.
       - sustain_bonus (3/step, per-step): tiny gradient to stay on pad until
         natural termination. 3 vs old 65 = 22x weaker over-stay incentive.
    2. All other components unchanged from v8 proven backbone.

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
    # Weight 2.0 (v6/v8 proven): consistent approach gradient.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6/v8 proven).
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6/v8 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: transition-based landing reward ---
    # KEY CHANGE (v9): switch from per-step accumulation to event-based.
    #
    # Problem: per-step bonus created over-stay exploitation.
    #   Old: +65 every step on pad → 3695 total, 99.7% magnitude share.
    #
    # Solution: one-time landing bonus + tiny sustain.
    #   - landing_bonus: rewards the TRANSITION into safe contact (first touch).
    #     Concentrated 500-point signal — strong enough to drive landing learning.
    #   - sustain_bonus: 3/step for maintaining contact. Provides gentle gradient
    #     to stay on pad until natural termination, but 22x weaker than old 65/step
    #     so over-stay is no longer the dominant strategy.
    #
    # Safe contact definition: both legs on pad, near target center.
    w_land = 500.0      # one-time landing bonus
    w_sustain = 3.0     # per-step sustain (was 65 in v8 → 22x reduction)

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
