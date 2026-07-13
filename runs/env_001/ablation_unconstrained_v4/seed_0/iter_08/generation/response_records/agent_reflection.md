# Response Record

## 分析

### 1. 这个 agent 发生了什么？

**行为诊断：hover exploit（悬停剥削）**

关键证据：
- **全部 20 episode 被 truncate（1000步）**，无一 terminate —— agent 从未真正着陆，只是活着
- `proximity_reward`（43.5%）和 `stable_landing_bonus`（51.8%）合计占奖励量级的 **95%+**，两者几乎每步都在支付（active_rate 100% 和 96.5%）
- `progress_delta` 仅占 0.6% —— 驱动 agent 靠近目标的信号被完全淹没
- `distance_penalty` 均值仅 -88，回推平均距离约 0.29 —— agent **确实靠近了目标**，但选择悬停收集 proximity_reward 的持续收益（每步 ~1.4），而不是着陆终止回合

本质上，`proximity_reward`（仅基于距离的连续支付）创造了一个"靠近但不降落"的局部最优：agent 在目标附近悬停 1000 步收集奖励，远比着陆后终止更"划算"。

### 2. 哪个组件最值得干预？

**`proximity_reward` —— 悬停激励的根源。** 它不要求任何着陆条件（速度、姿态、接触），仅凭距离就持续支付。移除此组件是消除 hover exploit 的必要条件。同时需要：
- 大幅提升 `progress_delta` 权重，使其成为驱动靠近的主信号
- 将 `stable_landing_bonus` 重构为**真正稀疏的着陆奖励**（严格门控：只有同时满足近距、低速、直立、双腿接触时才支付）

### 3. 我之前改了什么？

Iter 6 将 iter 2 的 `settle_reward`（稀疏着陆奖励）拆成了 `proximity_reward`（始终活跃的距离支付）+ `stable_landing_bonus`（软性稳定因子）。这个拆分制造了 hover exploit：proximity_reward 无条件支付，导致 agent 只需靠近就能持续获利，不再需要真正着陆。**必须回退这个拆分——移除 proximity_reward，让着陆奖励重新变得稀疏。**

---

现在执行修改：以 best（iter 2）的结构为参考，移除 proximity_reward，强化 progress_delta，重构 stable_landing_bonus 为严格门控的稀疏着陆奖励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes iter 6/7 hover exploit by:
    1. Removing proximity_reward — the always-on distance payment that incentivizes hovering
    2. Boosting progress_delta as the primary approach driver
    3. Redesigning stable_landing_bonus as a STRICT landing-only bonus:
       - Requires ALL of: near target, low speed, upright angle, both leg contacts
       - Multiplicative gating (product of independent conditions) ensures bonus ≈ 0
         unless every landing condition is simultaneously met
       - Sparse but high-value: pays meaningfully only during actual settling/landing
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (used ONLY as a modulator, never standalone) ---
    k_prox = 5.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (PRIMARY approach driver)
    # Boosted to compensate for removal of proximity_reward.
    # This must be the dominant positive signal driving the agent to the target.
    # ============================================================
    w_progress = 40.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (global baseline pull toward origin)
    # ============================================================
    w_dist = 0.5
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: stable_landing_bonus (STRICT landing-only sparse reward)
    # Multiplicative product of independent landing conditions.
    # Each factor ranges [0,1]; product is near-zero unless ALL are satisfied.
    # This replaces both proximity_reward and the old permissive bonus.
    # ============================================================
    # Independent landing-condition gates (each [0,1], 1 = perfect)
    near_target = max(0.0, 1.0 - curr_distance / 0.4)       # dist < 0.4
    speed_low = max(0.0, 1.0 - curr_speed / 0.3)            # speed < 0.3
    angle_upright = max(0.0, 1.0 - abs(nangle) / 0.25)     # |angle| < 0.25 rad
    both_contacts = nleft_contact * nright_contact           # 1.0 only if both legs touch

    w_stable = 25.0
    # Product gating: bonus > 0 only when ALL conditions are substantially met.
    # (0.5 + 0.5*both_contacts) ensures contact bonus is partial with one leg,
    # full with both — but still requires near/slow/upright to matter.
    landing_quality = near_target * speed_low * angle_upright * (0.3 + 0.7 * both_contacts)
    stable_landing_bonus = w_stable * landing_quality

    # ============================================================
    # Component D: velocity_damping (mild, proximity-gated speed penalty)
    # ============================================================
    w_vel = 0.3
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component E: orientation_penalty (mild, proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + stable_landing_bonus
        + velocity_damping
        + orientation_penalty
        + engine_penalty
    )

    reward_components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "stable_landing_bonus": float(stable_landing_bonus),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), reward_components
```
