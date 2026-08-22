Now I have a comprehensive understanding of the full history. Let me compile the diagnosis.

---

## 诊断报告

### Evidence

**历史轨迹（contact_bonus 为核心线索）：**

| iter | contact_w | contact ep_sum | contact active | score | 结论 |
|---|---|---|---|---|---|
| 1 | sparse | 0 | 0% | -32 | 从未着陆 |
| 2 | **100** | 2980 | 11.8% | 73 | ✅ 首次成功，但 contact 占 99.6% |
| 3 | ~squared pot | 35 | 0% | -50 | ❌ 试图削弱 contact → 着陆行为崩溃 |
| 4 | one-time +200 | 142 | 1.8% | -110 | ❌ dive→grab→die，len=68 |
| 5 | **25** | 0 | 0% | -96 | ❌ contact 太弱，从未激活 |
| 6 | **80** | 4548 | 13.7% | **154** | ✅ **史上最佳**，cp80% 达 240 |
| 7 | **40** | 32 | 0.1% | -121 | ❌ contact 崩塌，v6→v7 改动过猛 |

**v6 checkpoint 趋势（关键证据）：**
- Score: cp40%=70.9 → cp60%=149 → **cp80%=240.2** → cp100%=160.2
- contact_bonus: cp40%=9816 → cp60%=6880 → **cp80%=3888** → cp100%=4540
- goal_proximity: cp40%=2.30 → cp60%=2.51 → **cp80%=2.703** → cp100%=2.242

**v7 checkpoint 趋势（失败的证据）：**
- Score 全程在 -150 到 +14 之间震荡，从未稳定
- contact_bonus 剧烈波动：0 → 1258 → 8 → 762 → 32，无稳定增长
- 与 v6 的单调增长形成鲜明对比

### Behavior Diagnosis

**当前策略（v7）学到的行为**：agent 丧失了着陆能力。contact_bonus active_rate 仅 0.1%（vs v6 的 13.7%），ep_sum 从 4548 暴跌至 32。agent 在目标附近徘徊（len=772）但不落地。根本原因不是"接触奖励过大导致剥削"，而是 v7 将 contact 从 80 砍到 40 的同时将 velocity_penalty 从 0.3 提到 0.5、potential 从 2.0 提到 6.0，导致接近目标时的净奖励不足以激励着陆——agent 宁愿在远处徘徊。

**v6 的成功与退化**：v6 的架构是正确的——potential-based progress + per-step contact。cp80% 模型达到了 240（超过目标 200），证明了该架构的能力。退化发生在 cp80%→cp100%：agent 学会在平台上多停留（contact 3888→4540，+8 步），但接近效率下降（goal_proximity 2.703→2.242），导致总评分从 240 降至 160。

**核心矛盾**：历史反复证明，contact weight 必须 ≥80 才能可靠激活着陆行为（iter_2=100 成功，iter_6=80 成功，iter_5=25 死亡，iter_7=40 死亡）。但 contact=80 在 v6 中占 99.7% magnitude_share，导致 late-training 退化。

### 修改方案

**L1 微调**：回到 v6 已验证的架构，只做一个保守改动——contact 从 80 降至 65。这是一个温和的 19% 削减，远小于 v7 的 50% 砍半。历史上 40 崩溃、80 成功，65 处于安全区间但足以减弱过度停留的激励。其余参数保持 v6 原值（potential=2.0, velocity=0.3, orientation=0.05），因为这些值在 cp80% 时已被证明能产生 240 分的行为。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: v6 backbone with modest contact reduction (80→65) to curb late-training
    over-stay degradation while staying safely above the activation threshold.

    Diagnosis from v7 (score=-121, total regression from v6's 154):
    - contact 80→40 collapsed landing behavior (active 13.7%→0.1%).
      History shows contact must be ≥80 to reliably activate (iter_2=100✓,
      iter_6=80✓, iter_7=40✗, iter_5=25✗).
    - velocity 0.3→0.5 + potential 2→6 created conflicting near-ground signals
      that made approach net-positive but landing net-negative vs staying aloft.

    Diagnosis from v6 (best=154, cp80%=240, cp100%=160):
    - Architecture is sound: potential-based progress + per-step contact.
    - cp80%→cp100% degradation: agent over-stayed on pad (+8 contact steps),
      approach efficiency dropped (goal_proximity 2.703→2.242).
    - Contact at 99.7% magnitude_share drowns efficiency signals in late training.

    Changes from v7 (effectively reverting toward v6):
    1. w_contact: 40 → 65 — modest reduction from v6's 80 (19% cut, not 50%).
       Still well above the failure threshold seen at 40 and 25.
       Reduces per-step incentive to over-stay without killing landing motivation.
    2. w_pot: 6.0 → 2.0 — revert to v6's proven approach gradient.
       v7's 6.0 created conflicting incentives near ground.
    3. w_v: 0.5 → 0.3 — revert to v6's velocity damping.
       v7's 0.5 was too punitive near ground, discouraging final approach.
    4. w_ang: 0.08 → 0.05 — revert to v6's orientation weight.

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
    # Weight 2.0 (v6 proven): provides consistent approach gradient
    # without conflicting with contact incentive near ground.
    w_pot = 2.0
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    goal_prox = w_pot * (dist_cur - dist_nxt)

    # --- 2. velocity_penalty: quadratic, gated by proximity to pad ---
    # Dampens high-speed approaches near the ground. Gate keeps far-field
    # maneuvering unpenalized. Weight 0.3 (v6 proven): enough to deter
    # high-speed crashes without discouraging final approach.
    w_v = 0.3
    k_gate = 1.0
    gate = 1.0 / (1.0 + k_gate * (abs(x_nxt) + abs(y_nxt)))
    vel_penalty = -w_v * gate * (vx_nxt ** 2 + vy_nxt ** 2)

    # --- 3. orientation_penalty: posture stability ---
    # Penalize tilt and angular velocity to keep lander upright.
    # Weights unchanged from v6 — stable across all iterations.
    w_ang = 0.05
    w_angvel = 0.005
    orient_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # --- 4. contact_bonus: per-step reward for safe pad contact ---
    # Per-step design (v6/v2 proven): rewards every step with both legs
    # on pad near target. Weight 65: modest reduction from v6's 80 to
    # reduce late-training over-stay incentive while staying safely above
    # the ~40-50 threshold where landing behavior collapses.
    w_contact = 65.0
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