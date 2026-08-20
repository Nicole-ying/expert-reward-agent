# Response Record

现在让我进行逐组件分析：

---

## 逐组件诊断

### 1. `contact_guidance` — **主导过度 (98.1% magnitude_share)**

| 指标 | iter 1 | iter 2 | 趋势 |
|---|---|---|---|
| ep_sum_mean | 252.9 | 175.6 | ↓ (系数减半) |
| active_rate | 68.0% | 73.9% | ↑ |
| magnitude_share | 99.2% | 98.1% | → 几乎不变 |

**诊断**：上一轮将系数从 0.3 降到 0.15，score 从 171→206，方向正确。但 magnitude_share 仅从 99.2% 降到 98.1%（几乎没变），说明 agent 完全围绕 contact_guidance 优化——它会自适应地调整行为来最大化这个信号，系数变化主要影响 reward scale 而非相对组成。checkpoint 趋势显示 cp60%-cp80% 剧烈波动（133→85→17→169），训练不稳定。

**结论**：继续沿同一方向（降 contact、升 progress），因为上一轮此方向效果良好。

### 2. `progress` — **信号太弱 (1.2% share)**

| 指标 | iter 1 | iter 2 |
|---|---|---|
| ep_sum_mean | 0.65 | 2.08 |
| active_rate | 99.9% | 99.3% |

progress 几乎始终活跃但幅度极小。它对 agent 行为几乎没有塑造力——agent 靠近目标时获得的小额 progress 奖励完全被 contact_guidance 淹没。

### 3. `orientation_penalty` & `velocity_penalty` — **正常但微弱**

两个惩罚项都接近 100% active，幅度合理（-1.13 和 -0.18），在后台起约束作用。保持不动。

---

## 决策

- **问题**：contact_guidance 仍然极度主导（98.1%），progress 几乎无影响力
- **层级**：L1（score=205.7 > target=200，差距仅 2.85%，必须保守微调）
- **语义方向**：沿 iter 1→2 的成功方向，进一步从 contact 向 progress 再平衡
- **操作**：contact_guidance 系数 0.15→0.12，progress 系数 1.5→1.8

两个系数调整服务于同一个语义目标（"平衡 contact 和 progress 的相对权重"），属于单次语义修改。

---

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
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    velocity_ok = max(0.0, 1.0 - abs(vy) / 0.5)
    contact_guidance = 0.12 * contact_val * proximity_factor * velocity_ok

    # combine all components
    total_reward = (1.8 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 1.8 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```
