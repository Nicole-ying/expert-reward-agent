# Response Record

## 分析

### 1. Agent 发生了什么？
**所有20个评估episode全部早期坠毁**（terminated=20/20, len=68.45, score=-115）。行为证据：
- `progress_delta` 贡献 +44.79（100% active），是绝对主导信号
- `distance_penalty` -33.27，`velocity_damping` 仅 -5.17
- progress_delta 远超所有惩罚之和，agent 学到的是"全速冲向目标"而非"减速着陆"
- `stable_landing_bonus` 激活率仅 0.6%——agent 从未活到着陆阶段就已坠毁
- 结论：progress_delta 权重过大（w=40），agent 被驱动高速接近目标，没有足够减速激励，早期坠毁

### 2. 哪个组件最值得干预？
**progress_delta**（w=40）是核心问题。它占总 signed magnitude 的 50%+，比 velocity_damping 强近 9 倍。agent 正确学到了"接近目标"，但没学到"减速"。需要大幅降低 progress_delta 权重，并引入有效的近目标减速激励和更可达到的着陆信号。

**stable_landing_bonus** 是次要问题：0.6% active_rate 说明严格乘积门控在 agent 到达着陆区之前完全沉默，无法提供学习梯度。

### 3. 我之前改了什么？
Iter 7→8：移除了 `proximity_reward`（为避免 iter 6 的悬停漏洞），同时将 progress_delta 从较小值提升到 w=40 作为"主要接近驱动"。实际效果：agent 比 iter 6 更激进地冲向目标，62步坠毁 vs 以前 1000步悬停，score 从 +16 暴跌到 -115。**移除 proximity_reward + 过度提升 progress_delta 导致 crash**。

### 修改方案
基于 best（iter 2）的 settle_reward 成功模式：
1. **progress_delta 权重从 40→12**：恢复为温和引导信号，不再是支配力量
2. **用 settle_reward 替代 strict landing_bonus**：proximity-gated 加和式着陆奖励（各条件独立贡献），不要求所有条件同时完美满足
3. **distance_penalty 从 0.5→0.2**：减轻持续负向压力
4. **speed_penalty 加强近目标惩罚**：`w_speed = 0.5 + 3.0*prox`，远距温和、近距严厉，直接反制"冲过头"
5. 保留 engine_penalty、orientation_penalty 作为轻度正则

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes iter 8 crash-by-overspeed by:
    1. Reducing progress_delta from 40→12 — guide, not dominate
    2. Replacing strict multiplicative landing_bonus (0.6% active) with 
       proximity-gated settle_reward using additive condition factors
    3. Distance penalty reduced 0.5→0.2
    4. Speed penalty strengthened near target: w_speed = 0.5 + 3.0*prox
       to directly incentivize deceleration during final approach
    Based on iter 2 settle_reward success pattern.
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (modulator only) ---
    k_prox = 5.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (MODEST approach guide)
    # Reduced from 40→12. Must not dominate over landing incentives.
    # ============================================================
    w_progress = 12.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (mild baseline pull)
    # Reduced from 0.5→0.2 to prevent penalty stack overwhelming
    # ============================================================
    w_dist = 0.2
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: settle_reward (proximity-gated additive landing reward)
    # Replaces the strict multiplicative product (0.6% active).
    # Each landing condition independently contributes reward when near target.
    # near_factor is the master proximity gate; within it, slow/upright/contact
    # each add their own bounded contribution.
    # ============================================================
    near_factor = max(0.0, 1.0 - curr_distance / 0.5)        # within 0.5 units
    slow_factor = max(0.0, 1.0 - curr_speed / 0.4)           # speed < 0.4
    upright_factor = max(0.0, 1.0 - abs(nangle) / 0.3)       # |angle| < 0.3 rad
    contact_sum = nleft_contact + nright_contact              # 0, 1, or 2

    w_settle = 20.0
    # Additive within proximity gate: each factor contributes independently
    # so partial landing progress is rewarded, not just perfect completion.
    settle_quality = near_factor * (
        0.4 * slow_factor +
        0.3 * upright_factor +
        0.8 * min(contact_sum / 2.0, 1.0)
    )
    settle_reward = w_settle * settle_quality

    # ============================================================
    # Component D: speed_penalty (proximity-strengthened)
    # Mild far from target (0.5), strong near target (up to 3.5).
    # Directly incentivizes deceleration during final approach.
    # ============================================================
    w_speed_base = 0.5
    w_speed_prox = 3.0
    w_speed = w_speed_base + w_speed_prox * curr_prox
    speed_penalty = -w_speed * curr_speed

    # ============================================================
    # Component E: orientation_penalty (mild regularizer)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + settle_reward
        + speed_penalty
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "settle_reward": float(settle_reward),
        "speed_penalty": float(speed_penalty),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components
```
