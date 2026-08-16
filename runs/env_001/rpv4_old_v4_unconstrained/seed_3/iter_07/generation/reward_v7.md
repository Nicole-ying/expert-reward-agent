---

## 诊断报告

### evidence

**全局演变（6 轮）：**

| iter | score | len | term | 核心正信号 | 核心问题 |
|-----:|------:|----:|:----:|----------|---------|
| 1 | -32.4 | 1000 | 0/20 | 无 | 从未着陆 |
| 2 | 79.8 | 316 | 20/20 | per-step contact (92% share) | 着陆质量低，奖励被单一信号劫持 |
| 3 | 129.7 | 725 | 9/20 | one-shot contact (60%) + landing_approach (38%) | 悬停拖延着陆 |
| 4 | 10.3 | 788 | — | 移除 contact → 崩溃 | contact 是唯一着陆驱动力 |
| **5** | **175.7** | 589 | 16/20 | contact_bonus (78%) + settled (20%) | **cp80-90% 出现 bounce exploit** |
| 6 | 57.6 | 846 | 7/20 | **pre_landing_bonus (62%)** | **悬停不降落，严重退化** |

**Iter 5 → Iter 6 退化的根本原因：**

Iter 5 的 checkpoint 趋势揭示了 bounce exploit：
- `contact_bonus` 在 cp80% 飙到 2454、cp90% 飙到 2711（active 10.5%），同时 score 从 cp50% 的 249 暴跌到 cp80% 的 114
- 原因：agent 学会了微小跳跃（barely lift off → re-contact），以近乎完美的 gates（spatial_gate ≈ 0.96, speed_gate ≈ 0.32）反复触发 one-shot bonus，每次约 15-16 点，累积数百次

Iter 6 的"修复"（引入 `pre_landing_bonus`）制造了更严重的问题：
- `pre_landing_bonus` ep_sum=186.8（62% share），active_rate=96.8%——成为绝对主导信号
- 该组件有 `not_contacted = 1.0 - has_contact` 门控：**接触时关闭，不接触时激活**
- `settled_reward` 从 iter5 的 297 暴跌到 27.3（3.2% active vs 38.3%）
- `contact_bonus` 从 1177 暴跌到 2.2（0.1% active vs 5.8%）
- **只有 7/20 episode 终止**（其余 13 个跑满 1000 步被截断）
- `goal_proximity` 从 -9.9 恶化到 -63.8——agent 离目标更远

### behavior_diagnosis

**当前策略（iter 6）学到的是"悬停寄生"而非着陆：**

agent 发现了一个局部最优：在着陆平台正上方悬停但永不接触。`pre_landing_bonus` 每步给约 0.2-0.4 点（w=0.40 × near_pad × pre_slow × pre_upright），持续 846 步累积 187 点。而着陆后的 `settled_reward`（约 2-3/步）虽然更高，但 episode 着陆后很快终止（body_not_awake_or_settled），agent 实际体验到的 settled 总回报远低于悬停的确定性收益。本质上，`pre_landing_bonus` 创造了一个"接近但不接触"的奖励陷阱。

**核心矛盾**：`pre_landing_bonus` 的 `not_contacted` 门控使得"完成着陆"意味着丧失最大收入来源。这是结构性设计错误。

### 干预策略

1. **删除 `pre_landing_bonus`**：它是悬停陷阱的根源。
2. **删除 `contact_break_penalty`**：iter 6 证据显示它几乎从未激活（active 0.0%，ep_sum=-1.2），对抑制 bounce 无效。
3. **恢复 iter 5 的 contact_bonus 驱动架构**：这是唯一被证明能产生着陆行为的信号。
4. **新增 `approach_scale` 反 bounce 机制**：利用 `dist_cur`（接触前距离）缩放 contact_bonus——从远处靠近的 genuine landing 获得全额奖励，从近处微小跳跃的 re-trigger 被压制到 ~10%。
5. **大幅强化 `settled_reward`**：w 从 iter5 的 2.0 → 6.0，同时放宽 gates（σ² 从 0.10/0.08/0.15 → 0.25/0.15/0.30），使稳定停靠的每步收益明确碾压任何 bounce 策略。
6. **简化 `goal_proximity`**：线性距离惩罚，不加 approach_bonus（避免振荡）。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: return to contact_bonus-driven architecture (proven in iter_5, score=175.7)
        with structural anti-bounce fixes.

    Diagnosis (from iter_5→6 evidence):
    - iter_5 succeeded with one-shot contact_bonus (w=50) + settled_reward (w=2),
      but cp80-90% showed bounce exploit: agent learned tiny hops to re-trigger
      contact_bonus at near-perfect gates → 2454-2711 ep_sum, score crashed.
    - iter_6 tried pre_landing_bonus to bridge approach→landing, but the
      not_contacted gate created a hover trap: agent hovered indefinitely near
      pad (96.8% active, 62% share) without landing. Only 7/20 terminated.
    - contact_break_penalty was a dead letter (0.0% active, -1.2 ep_sum).

    Key changes from v6:
    1. REMOVED pre_landing_bonus — the hover trap root cause.
    2. REMOVED contact_break_penalty — never activated, no evidence of efficacy.
    3. contact_bonus: w 15→40, ADDED approach_scale: bonus scales with how far the
       agent approached FROM. Genuine landing from distance → full bonus. Tiny hop
       re-trigger (dist_cur ≈ 0.05) → ~10% bonus. This structurally prevents bounce.
    4. settled_reward: w 4→6, wider gates (spatial σ² 0.10→0.25, speed 0.08→0.15,
       angle 0.15→0.30). Makes sustained contact clearly dominant (~5/step vs ~0.2
       for any residual bounce).
    5. contact_speed_penalty: w 1.5→2.0 — stronger speed suppression while contacted.
    6. goal_proximity: keep linear distance (same as v6), no approach_bonus.
    7. velocity_penalty: keep gated + anti-slam (same as v6).
    """

    # ── unpack ──────────────────────────────────────────────────────
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_c_cur, right_c_cur = obs[6], obs[7]
    left_c_nxt, right_c_nxt = next_obs[6], next_obs[7]

    # ── derived ─────────────────────────────────────────────────────
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — linear distance penalty
    #    Simple, monotonic, consistent gradient. No approach_bonus (avoids
    #    oscillation: approach→overshoot→re-approach cycles).
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.25
    goal_proximity = -w_dist * dist_nxt               # -1.25 at dist=5, -0.25 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + anti-slam near ground
    #    Gating allows free motion far from pad. Anti-slam harshly penalises
    #    downward velocity when very close to ground (prevents crash-landing).
    # ══════════════════════════════════════════════════════════════════
    k_gate = 4.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 at pad, ~0.2 at dist=1

    w_v = 0.40
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    w_slam = 3.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.03)    # tight: only dist < ~0.2 matters
    downward_v = max(0.0, -vy_nxt)
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.06
    w_angvel = 0.015
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — micro-shaping, anti-hover whisper
    #    Weight kept tiny (0.03). Provides soft gradient toward slow+near+upright
    #    without creating a profitable steady state.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.03 * near_y * near_x * slow * upright

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus with anti-bounce gate
    #
    #    ANTI-BOUNCE MECHANISM: approach_scale
    #    The bonus is scaled by 1 - exp(-dist_cur / 0.50), where dist_cur is
    #    the agent's distance from pad BEFORE making contact.
    #    - Genuine landing from far away (dist_cur=3.0): scale ≈ 0.998 → full bonus
    #    - Tiny hop re-trigger (dist_cur=0.05): scale ≈ 0.095 → only ~10% bonus
    #    This structurally prevents the bounce exploit seen in iter_5 cp80-90%.
    #    A bounce cycle nets ~3-4 bonus minus penalties → clearly inferior to
    #    settled_reward's ~5-6 per step.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 40.0
    new_contact = has_contact * (1.0 - had_contact)       # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.04)   # σ²=0.04: tight, at y=0.05 gate=0.94
    speed_gate   = 2.718281828 ** (-speed_sq / 0.04)      # σ²=0.04: tight, at speed=0.3 gate=0.11
    approach_scale = 1.0 - 2.718281828 ** (-dist_cur / 0.50)  # anti-bounce: near→low, far→high

    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate * approach_scale

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #
    #    BOOSTED: w 4→6, wider gates. The dominant positive signal when landed.
    #    ~6 * 0.96 * 0.94 * 0.97 ≈ 5.25/step when well-settled.
    #    Over 200 settled steps ≈ 1050 — dwarfs any residual bounce profit.
    #    Wide gates ensure agent can discover this reward even with imperfect landing.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 6.0
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.25)    # σ²=0.25: tolerant
    s_slow     = 2.718281828 ** (-speed_sq / 0.15)       # σ²=0.15: tolerant
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30) # σ²=0.30: tolerant
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce: punish any speed while contacted
    #    Strengthened from 1.5→2.0. Even a tiny bounce requires lifting off,
    #    which incurs speed while (briefly) in contact → penalty.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 2.0
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 8. fuel_penalty — small regularisation
    # ══════════════════════════════════════════════════════════════════
    w_fuel = 0.005
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ══════════════════════════════════════════════════════════════════
    # assemble
    # ══════════════════════════════════════════════════════════════════
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
```