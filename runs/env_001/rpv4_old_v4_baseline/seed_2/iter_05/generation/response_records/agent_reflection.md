# Response Record

## 逐组件诊断

### 1. `contact_guidance` — 仍占主导（69.1%），但趋势健康

| 指标 | iter 1 | iter 2 | iter 3 | iter 4 | 趋势 |
|---|---|---|---|---|---|
| ep_sum_mean | 252.9 | 175.6 | 133.2 | 12.0 | ↓ 持续下降 |
| magnitude_share | 99.2% | 98.1% | 97.2% | 69.1% | ↓ iter 4 大幅突破 |
| active_rate | 68.0% | 73.9% | 70.1% | 28.0% | ↓ |

**Checkpoint 趋势**：cp70% 达峰 36.19 → cp100% 降至 15.0，active_rate 从 54%→32%。agent 在训练后期自然减少对 contact 的依赖——这是健康信号，不是 exploit。

**诊断**：iter 4 将 proximity 从 `dist/5.0` 缩到 `dist/3.0` 是关键突破——contact ep_sum 从 133 骤降至 12（10倍），len 从 798 降至 343。但 69.1% magnitude_share 仍然偏高，说明 agent 仍围绕 contact 信号驱动。可以继续沿此方向微调。

### 2. `progress` — 健康增长，占比改善

| 指标 | iter 1 | iter 4 |
|---|---|---|
| ep_sum_mean | 0.65 | 3.28 |
| magnitude_share | 0.3% | 20.4% |

Checkpoint 趋势：1.90 → 3.27，全程稳步上升。progress 系数从 0.5→1.5→1.8→2.5 每次提升都改善了平衡。方向正确。

### 3. `orientation_penalty` — 稳定工作

ep_sum=-1.587，share=9.2%，全程 100% active。cp60% 达峰 -2.38 后改善至 -1.68。正常工作，无需修改。

### 4. `velocity_penalty` — 稳定且收敛

ep_sum=-0.246，share=1.4%。训练中持续改善（-0.553→-0.254），agent 学会了控制速度。无需修改。

---

## 修改方案：L1 — 继续收缩 contact，提升 progress

**语义方向**：延续 iter 4 已验证的方向——缩小 contact 窗口 + 降低 contact 系数 + 提升 progress 权重。三个参数调整服务于同一目标：将奖励重心从"状态型接触奖励"进一步转移到"过程型导航进步"。

- `contact_guidance` 系数：0.07 → **0.05**
- `proximity_factor` 窗口：`dist / 3.0` → **`dist / 2.5`**
- `progress` 系数：2.5 → **3.0**

均为 L1 级调参，不改变数学形态。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # distance to target (target at origin (0,0))
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # velocity components
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # orientation penalty: body angle and angular velocity
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # contact guidance: reward both legs touching, but only near the target and at low speed
    contact_val = float(next_obs[6] + next_obs[7])   # range 0.0 .. 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 2.5)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.05 * contact_val * proximity_factor * velocity_ok

    # combine all components
    total_reward = (3.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 3.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```
