## 逐组件诊断

### 1. `contact_guidance` — 主导过度（99.2% magnitude_share）

**证据**：
- ep_sum=252.86，占 magnitude_share 的 99.2%，其他三个组件合计不到 1%
- checkpoint 趋势：cp60%=0 → cp70%=248（突破点）→ cp100%=167（逐渐下降）
- 最终 eval 回升到 252.86，说明 agent 在平台上滞留时间长，持续收割 contact 奖励
- 仅 7/20 成功 terminated，13/20 被 truncate，len=781 — agent 着陆后迟迟不触发 settled 终止

**诊断**：这个组件是 agent 学会着陆的关键信号（cp60% 之前为 0 时 score 为负），但它过于强势。agent 一旦上平台就持续领奖，没有动力快速完成 settled。属于「占据好状态即可持续获奖」模式，但因为是 L1 层级，只能调系数。

### 2. `progress` — 信号过弱（0.3% share）

**证据**：ep_sum=0.65，几乎被 contact_guidance 淹没。checkpoint 全程在 0.19~0.67 窄幅波动。99.9% active 说明每步都在计算，但量级太小，无法引导高效的导航轨迹。

**诊断**：progress 应该是导航阶段的主信号，但权重太低，agent 基本感受不到。需要提升权重来加速接近目标。

### 3. `orientation_penalty` — 正常运行（0.5% share）

ep_sum=-1.20，active_rate=100%，checkpoint 全程稳定在 -0.5~-1.7。无需改动。

### 4. `velocity_penalty` — 过弱但方向正确（0.1% share）

ep_sum=-0.18，checkpoint 从 -0.49 逐渐收敛到 -0.18，说明 agent 在学习减速。量级太小无法约束行为，但考虑到当前主要矛盾是 contact_guidance 的压倒性主导，velocity 暂不调整，避免引入过多变量。

---

## 决策

**L1 修改**：只调系数。两个调整服务于同一语义——**重新平衡 contact_guidance 与 progress 的量级关系**，让 agent 在保持着陆能力的同时，获得更强的导航效率信号。

| 参数 | 旧值 | 新值 | 理由 |
|---|---|---|---|
| contact_guidance 系数 | 0.3 | 0.15 | 减半，预期 ep_sum ~126，仍足够维持着陆行为 |
| progress 系数 | 0.5 | 1.5 | 三倍，预期 ep_sum ~1.95，增强导航引导 |

预期 contact_guidance 的 magnitude_share 从 99% 降到 ~95%，progress 从 0.3% 升到 ~2%。着陆行为保留，同时 agent 有更强动力高效接近目标、尽快触发 settled。

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
    contact_guidance = 0.15 * contact_val * proximity_factor * velocity_ok

    # combine all components
    total_reward = (1.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 1.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
    }
    return total_reward, components
```