# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 1. Progress delta: reward for reducing distance to goal ---
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    raw_progress = dist - dist_next
    # Clip to keep the signal bounded and avoid huge single‑step rewards
    progress = max(-2.0, min(2.0, raw_progress))

    # --- 2. Stability cost with altitude-dependent weighting ---
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    alt = next_obs[1]  # relative height above landing pad (goal y = 0)

    # The closer to the ground, the stronger the requirement for low speed / angle.
    altitude_factor = 1.0 + 2.718281828 ** (-alt / 0.5)  # e^(-2*alt), approx 1 far away, up to 2 near ground

    stability_cost = (0.2 * abs(vx) +
                      0.2 * abs(vy) +
                      0.2 * abs(angle) +
                      0.2 * abs(angvel)) * altitude_factor

    # --- Combine ---
    total_reward = 10.0 * progress - stability_cost

    components = {
        "progress_delta": 10.0 * progress,
        "weighted_stability_penalty": -stability_cost
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 组件与角色

- **主学习信号 – `progress_delta`**  
  每步奖励向目标移动的量：`dist(obs) - dist(next_obs)`。它是一个稠密的、逐步可微的信号，直接告诉 agent“靠近目标是有益的”。相比常负的 `distance_reward`（之前尝试并停滞），`progress_delta` 更强调每步的改进，避免 agent 在目标附近“只拿负距离分而不真正完成”的陷阱。对原始进度做了 [-2, 2] 的裁剪，防止单步大跳产生异常数值。

- **稳定约束 – `weighted_stability_penalty`**  
  对水平/垂直速度、倾角、角速度的绝对值做轻量惩罚，乘以一个 **随高度变化的因子**。因子在高处约为 1，靠近地面的高度（`alt→0`）时逐渐升至 2，这样迫使 agent 在即将着陆时必须做到低速、竖直、无旋转。相较于之前各迭代中使用的固定权重 `stability_penalty`，**高度自适应** 的数学形态是一个本质上全新的设计假设，它为着陆阶段提供了稠密的减速引导，而不需要一个独立的离散 `soft_landing_bonus`（后者在早期暴露了触发过稀、易被利用的问题）。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境没有暴露出显式的 success/failure 标志（`explicit_success_flag_available=false`，`info` 恒为空），因此不能在 reward 中依赖它们。我们的设计完全基于可观测的连续信号（位置、速度、姿态），符合“信号可用性优先”原则。

## 哪些组件留到后续迭代

- **显式成功/失败奖励**：待环境 wrapper 提供可靠的 termination flag 后再加入。
- **能量/燃料效率惩罚**：`energy_penalty` 会抑制 agent 使用主引擎，在学会稳定着陆之前加入可能导致“不敢动作”；后续在任务成功率提升后再以小权重引入。
- **time_penalty**：加快进度，但先要保证安全完成。
- **复杂门控或动态课程**：在基础策略稳定后考虑精调。

## 训练后应观察的 failure mode

- **目标附近震荡**：`progress_delta` 可能在靠近后梯度变弱，agent 可能来回飘动；此时需要检查 `weighted_stability_penalty` 的着陆区抑制是否足够，考虑微调 altitude 尺度或系数。
- **高速俯冲后 crash**：如果稳定权重在远端的惩罚太弱，agent 可能以极高速度接近地面导致碰撞；可通过监测步均 `progress_delta` 与 `stability_cost` 的比值调整参数。
- **始终不完成着陆（hovering）**：若 altitude_factor 的强度不足以驱动最终减速，agent 可能悬停在目标上空却不触地；后续可考虑在接近地面时进一步放大惩罚，或加入一个温和的多条件软着陆奖励。
