# 设计理由
本轮是**正常模式**，旨在解决 `soft_landing` 完全失效（active rate 0%）且 agent 在 1000 步内徘徊不靠近目标的问题。核心诊断为：**静态距离奖励不足以驱动机器人进入着陆区**——`proximity_reward` 在远处仍有 ~0.8 的收益，agent 学会了停在一个中间距离“吃奖励”，缺少让距离持续缩小的驱动力。  
因此新增 **`progress_reward`**（Level 2 结构变换，添加一个组件），奖励每一步相对于上一步的距离缩小量（`delta_dist = dist_prev - dist_next`），用 `max(0, delta_dist)` 只给正向进步，系数设为 0.5（约为主信号 per‑step 的 0.58 倍）。  
该组件用连续、有界的形式（`0.5 * max(0, Δd)`），避免 iter 2 中过强 progress 导致的长度暴跌。原有 `proximity_reward` 和 `attitude_penalty` 保持不变，`soft_landing` 的门控暂时保留（一旦 agent 接近，其多个因子会提供精细稳定信号）。  
预期效果：每一步向目标靠近都会立即得到奖励，打破静态停留策略，驱使 agent 最终进入 `dist < 0.3` 的着陆区域，从而激活 `soft_landing`，完成着陆并恢复高分。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── Unpack observations ─────────────────────────────────────────
    px_prev, py_prev = obs[0], obs[1]          # previous position
    px, py = next_obs[0], next_obs[1]          # current position
    vx, vy = next_obs[2], next_obs[3]          # velocity
    angle  = next_obs[4]                       # body angle
    angvel = next_obs[5]                       # angular velocity
    left_leg  = next_obs[6]                    # left contact
    right_leg = next_obs[7]                    # right contact

    # ── Derived signals ─────────────────────────────────────────────
    dist_prev = (px_prev**2 + py_prev**2) ** 0.5
    dist_next = (px**2 + py**2) ** 0.5
    speed     = (vx**2 + vy**2) ** 0.5

    # ── 1. Proximity reward (static distance attractor) ─────────────
    proximity_reward = 1.0 / (1.0 + dist_next**2)

    # ── 2. Attitude penalty (unchanged) ─────────────────────────────
    attitude_penalty = -0.003 * (angle ** 2) - 0.001 * (angvel ** 2)

    # ── 3. Progress reward (NEW) ────────────────────────────────────
    # Reward each step that reduces distance to origin.
    delta_dist = dist_prev - dist_next          # positive when approaching
    progress_reward = 0.5 * max(0.0, delta_dist)

    # ── 4. Soft landing (gate kept, activates below 0.3) ────────────
    if dist_next < 0.3:
        contact_factor = (left_leg + right_leg) / 2.0
        speed_factor   = 1.0 / (1.0 + 10.0 * speed)
        angle_factor   = 1.0 / (1.0 + 5.0 * (angle ** 2))
        soft_landing   = contact_factor * speed_factor * angle_factor
    else:
        soft_landing = 0.0

    # ── Combine ─────────────────────────────────────────────────────
    total_reward = (
        1.0 * proximity_reward
        + 1.0 * attitude_penalty
        + 1.0 * progress_reward
        + 2.0 * soft_landing
    )

    components = {
        "proximity_reward": proximity_reward,
        "attitude_penalty": attitude_penalty,
        "progress_reward": progress_reward,
        "soft_landing": soft_landing,
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 所有可用的 8 维观测均已使用，但缺少“距离是否在缩小”的进步信号，导致 agent 在静态奖励平台上停滞。
- **behavior**: Agent 在安全距离（约 0.5‑1.0）徘徊 1000 步，永不靠近，依靠 `proximity_reward` 维持非负收益，`soft_landing` 从未激活。
- **signal**: 缺乏 **distance progress** 信号；现有静态距离奖励无法驱使最后一段接近。
- **level**: Level 2
- **hypothesis**: 新增 `progress_reward` 使每一步靠近都能获得即时奖励，打破静止策略，引导 agent 进入着陆门控区域，从而激活 `soft_landing`，恢复高分。
- **risk**: 若系数 0.5 偏大，可能导致急躁接近而 crash（类似 iter 2），但 0.5 远低于 iter 2 的进度权重，且主信号仍在安全范围内。