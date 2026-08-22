Now I have a thorough picture. Let me analyze what's happening.

---

## 诊断报告

### evidence

**全局演变（8 轮）：**

| iter | score | len | term | 主导正信号 | 关键问题 |
|-----:|------:|----:|:----:|----------|---------|
| 1 | -32.4 | 1000 | 0/20 | 无 | 从未着陆 |
| 2 | 79.8 | 316 | 20/20 | per-step contact (92%) | 着陆质量低 |
| 3 | 129.7 | 725 | 9/20 | one-shot contact (60%) + landing_approach (38%) | 悬停拖延 |
| 4 | 10.3 | 788 | — | 移除 contact → 崩溃 | — |
| 5 | 175.7 | 589 | 16/20 | contact_bonus (78%, ep=1177) | 弹跳循环(cp30-80%后自愈) |
| 6 | 57.6 | 846 | — | contact_break 杀死了contact | 回退 |
| 7 | 24.8 | 688 | — | 过度收紧 gate → contact 死 | 回退 |
| **8** | **229.8** | **278** | **19/20** | **settled_reward (76%, ep=222)** | **settled 过度主导，cp70%→100% 分数退化** |

**iter_8 checkpoint 趋势关键信号：**

- Score 曲线: cp10%=-369 → cp70%=**253.6** (峰值) → cp100%=**224.2** (退化 29.4 分)
- settled_reward: cp70%=130 → cp100%=**409.6** (暴涨 3x) — agent 在后期疯狂优化 settled_reward
- contact_bonus: cp30%=440.7 (爆发) → cp40%=1.3 (崩溃) → cp60-100% 恢复至 32。说明早期发现弹跳 exploit 后又重新学习正常着陆

**v8 最终策略成分分析：**

| 组件 | ep_sum | magnitude_share | active_rate | 评价 |
|------|--------|:---:|:---:|------|
| settled_reward | 222.0 | **76.0%** | 32.6% | ⚠️ 过度主导 |
| contact_bonus | 50.2 | 17.2% | 0.6% | ✅ 一次性，合理 |
| goal_proximity | -10.1 | 3.4% | 100% | ✅ 持续梯度 |
| velocity_penalty | -3.8 | 1.3% | 96.4% | ✅ |
| landing_approach | 3.6 | 1.2% | 100% | ✅ |
| others | — | <1% | — | ✅ |

### behavior_diagnosis

当前策略学到的行为模式：

1. **快速着陆 + 长时间逗留**：ep_len=278，19/20 terminated（成功着陆）。agent 学会了快速接近（~200 步）然后稳定停靠（~78 步，32.6% active_rate）。这是**健康的**。

2. **settled_reward 过度主导（76%）**：虽然策略目前有效（score=229 > target=200），但 checkpoint 趋势揭示了隐患——score 从 cp70% 的 253.6 **退化**到 cp100% 的 224.2，同时 settled_reward 从 130 **暴涨**到 409.6。这说明 agent 在训练后期**过拟合了 settled_reward**，牺牲了其他方面的表现（可能包括更高效的路径或更低的燃料消耗）。

3. **燃料效率偏低**：fuel_penalty active_rate=83.6%，意味着 83.6% 的步使用引擎。虽然 penalty 绝对值很小，但高引擎使用率暗示 agent 没有动力优化燃料。

4. **contact_bonus 相对弱势**：ep_sum=50.2 vs iter_5 的 1177。这不是坏事（iter_5 有弹跳 exploit），但当前 50.2 的 landing moment 奖励相对于 222 的 settled 奖励偏弱，可能导致 agent 不太关心着陆瞬间的质量。

### 干预策略

**核心思路：微调 rebalance，压低 settled_reward 主导地位，提升着陆质量和燃料效率。**

具体变更（相对 v8）：

1. **settled_reward w: 3.5 → 3.0**（-14%）— 直接削减其 magnitude，目标将其 share 从 76% 降至 ~65%。不收紧 gate（保持可发现性）。
2. **contact_bonus w: 50 → 60，speed gate σ²: 0.08 → 0.06** — 增强着陆瞬间奖励 + 收紧速度门控，要求更慢的着陆。
3. **fuel_penalty w: 0.005 → 0.01** — 加倍燃料惩罚，从 -1.16 → ~-2.3，仍远小于主要信号但提供更强的效率梯度。
4. **velocity_penalty w_v: 0.30 → 0.25** — 略微放松通用速度惩罚，让 agent 在接近阶段有更大机动自由度。slam 保护保持。
5. **contact_speed_penalty w: 1.8 → 2.0** — 微增 anti-bounce 摩擦。

这些是 10-20% 级别的调参，不改架构。v8 的核心框架已被证明有效，只做权重再平衡。

---

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v9: rebalance — reduce settled_reward dominance (76%→~65%), boost contact_bonus
        quality, and increase fuel efficiency pressure.

    Diagnosis (iter_8 → v9):
    - v8 score=229.8 (exceeds target 200), but checkpoint trend shows score peaked
      at cp70% (253.6) then declined to cp100% (224.2) while settled_reward surged
      from 130→410. This is classic over-optimization of a single component.
    - settled_reward at 76% magnitude_share dominates the reward landscape, crowding
      out signals for approach quality and fuel efficiency.
    - fuel active_rate=83.6% with negligible penalty (-1.16) — no real efficiency pressure.

    Changes from v8:
    1. settled_reward: w 3.5→3.0. Gates unchanged (proven discoverable).
    2. contact_bonus: w 50→60, speed gate σ² 0.08→0.06. Stronger landing signal
       with tighter speed requirement.
    3. fuel_penalty: w 0.005→0.01. Double efficiency pressure.
    4. velocity_penalty: w_v 0.30→0.25. Slightly looser general speed damping
       for more dynamic approaches.
    5. contact_speed_penalty: w 1.8→2.0. Modest anti-bounce hardening.
    6. landing_approach: w 0.04→0.05. Slightly stronger whisper.
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
    dist_nxt_sq = x_nxt ** 2 + y_nxt ** 2
    dist_nxt = dist_nxt_sq ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2
    had_contact = left_c_cur * right_c_cur          # 0.0 or 1.0
    has_contact = left_c_nxt * right_c_nxt           # 0.0 or 1.0

    # ══════════════════════════════════════════════════════════════════
    # 1. goal_proximity — squared-distance penalty
    #    Strong far-field gradient (at dist=5: -2.5), gentle near pad.
    #    No approach_bonus: avoids oscillation (approach→overshoot→re-approach).
    # ══════════════════════════════════════════════════════════════════
    w_dist = 0.10
    goal_proximity = -w_dist * dist_nxt_sq            # -2.5 at dist=5, -0.1 at dist=1

    # ══════════════════════════════════════════════════════════════════
    # 2. velocity_penalty — gated speed cost + anti-slam near ground
    #    w_v 0.30→0.25: slightly looser general damping for more dynamic approaches.
    #    Gate allows free motion far from pad.
    #    Anti-slam penalises downward velocity when very close to ground.
    # ══════════════════════════════════════════════════════════════════
    k_gate = 5.0
    proximity_gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 near pad, ~0.17 at dist=1

    w_v = 0.25                                            # v8: 0.30
    gated_speed_penalty = -w_v * proximity_gate * speed_sq

    w_slam = 2.0
    slam_gate = 2.718281828 ** (-dist_nxt_sq / 0.04)    # tight: only dist < ~0.25 matters
    downward_v = max(0.0, -vy_nxt)
    anti_slam = -w_slam * slam_gate * (downward_v ** 2)

    velocity_penalty = gated_speed_penalty + anti_slam

    # ══════════════════════════════════════════════════════════════════
    # 3. orientation_penalty — stay upright, suppress spin
    # ══════════════════════════════════════════════════════════════════
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ══════════════════════════════════════════════════════════════════
    # 4. landing_approach — whisper-level shaping
    #    w 0.04→0.05: slightly stronger whisper for approach guidance.
    # ══════════════════════════════════════════════════════════════════
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)
    slow     = 2.718281828 ** (-speed_sq / 0.60)
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.30)
    landing_approach = 0.05 * near_y * near_x * slow * upright   # v8: 0.04

    # ══════════════════════════════════════════════════════════════════
    # 5. contact_bonus — ONE-SHOT transition bonus
    #
    #    w 50→60: stronger landing moment signal.
    #    speed gate σ² 0.08→0.06: tighter, demands slower touchdown.
    #
    #    Fires ONCE on first touchdown. Spatial + speed gates ensure
    #    only clean landings near the pad get the full bonus.
    # ══════════════════════════════════════════════════════════════════
    w_contact = 60.0                                        # v8: 50.0
    new_contact = has_contact * (1.0 - had_contact)         # 1 ONLY on transition step

    spatial_gate = 2.718281828 ** (-dist_nxt_sq / 0.06)     # σ²=0.06: unchanged
    speed_gate   = 2.718281828 ** (-speed_sq / 0.06)        # σ²=0.06: v8 was 0.08

    contact_bonus = w_contact * new_contact * spatial_gate * speed_gate

    # ══════════════════════════════════════════════════════════════════
    # 6. settled_reward — per-step reward for STAYING landed
    #
    #    w 3.5→3.0: reduce dominance from 76% toward ~65%.
    #    Gates unchanged: proven discoverable and effective.
    #    Only active when both legs are in sustained contact.
    # ══════════════════════════════════════════════════════════════════
    w_settled = 3.0                                         # v8: 3.5
    s_spatial  = 2.718281828 ** (-dist_nxt_sq / 0.12)      # σ²=0.12: unchanged
    s_slow     = 2.718281828 ** (-speed_sq / 0.10)         # σ²=0.10: unchanged
    s_upright  = 2.718281828 ** (-abs(angle_nxt) / 0.20)   # σ²=0.20: unchanged
    settled_reward = w_settled * has_contact * s_spatial * s_slow * s_upright

    # ══════════════════════════════════════════════════════════════════
    # 7. contact_speed_penalty — anti-bounce friction
    #    w 1.8→2.0: moderate hardening.
    #    Any speed while in contact incurs penalty, making bounce cycles costlier.
    # ══════════════════════════════════════════════════════════════════
    w_contact_spd = 2.0                                     # v8: 1.8
    contact_speed_penalty = -w_contact_spd * has_contact * speed_sq

    # ══════════════════════════════════════════════════════════════════
    # 8. fuel_penalty — efficiency regularisation
    #    w 0.005→0.01: double efficiency pressure.
    #    Still small relative to main signals but provides meaningful gradient.
    # ══════════════════════════════════════════════════════════════════
    w_fuel = 0.01                                           # v8: 0.005
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