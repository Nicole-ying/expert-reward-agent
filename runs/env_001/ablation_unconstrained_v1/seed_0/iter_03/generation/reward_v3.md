## 分析

### 1. Agent 发生了什么？

**核心事实**：
- 所有 20 个 episode 都被截断在 1000 步——没有一次成功着陆或失败终止。
- `landing_proxy` active_rate = **0.0%**：乘积 `proximity * stillness * upright * contact_factor` 始终为 0。因为 agent 从未进入距离目标 0.3 以内（`proximity` 要求 `dist < 0.3`）。
- `progress_reward` 仅累积 +5.46（1000 步净前进约 0.55 单位），而 `time_penalty` 稳定贡献 -20.0（1000 × -0.02），占总负分的 69%。
- score = -84.69，所有 episode 徘徊在起始位置附近，无法靠近目标。

**根因**：势能塑形 `10*(dist_current - dist_next)` 虽然理论正确，但 agent 在离散动作空间下几乎无法产生一致的净前进——1000 步仅移动 ~0.55 单位。距离主导信号太弱，时间惩罚主导了总分，且 landing proxy 的阈值（distance < 0.3）对当前策略完全不可达。

### 2. 哪个组件最值得干预？

**距离主信号**。`progress_reward` 的势能塑形在理论上保证无偏，但它的梯度只反映"进步量"而非"当前状态好坏"。当 agent 反复振荡时，净进步趋近于零，梯度信号消失。相比之下，`1/(1+distance)` 形式的 bounded proximity 在**任意位置**都提供指向目标的梯度，且越近梯度越强。历史 best 代码用了这个结构，搜索结果 `anti_oscillation_potential` 也建议用 bounded proximity 替代纯 delta。

同时，`landing_proxy` 的阈值需放宽：distance 门槛从 0.3 提到 0.5，让 agent 学会接近目标后 landing 信号自然激活。

### 3. 我之前改了什么？

上一轮我把 best 的 `1/(1+distance)` + `stability_penalty` + 稀疏 `landing_bonus` 改成了 `10*progress_delta` + 连续 `landing_proxy` + `time_penalty`。结果是：
- progress_delta 信号太弱（净前进 ~0.55），
- landing_proxy 阈值过紧（始终为 0），
- time_penalty 主导总分为负。

**本轮修改**：以 best 结构为基础（`1/(1+distance)` + stability + landing），但将稀疏 hard landing bonus 替换为**连续软着陆乘积**（解决 sparse_completion_proxy 问题），放宽 proximity 阈值，并移除 time_penalty（它正阻碍学习）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Next state
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    
    # 1. Bounded proximity: always provides gradient toward target
    #    (0, 1], peaks at 1 when exactly on target
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    r_dist = 1.0 / (1.0 + distance)
    
    # 2. Stability penalty: discourage violent motion, promote gentle flight
    r_stability = -(
        0.01 * (abs(x_vel) + abs(y_vel)) +
        0.1 * abs(body_angle) +
        0.05 * abs(ang_vel)
    )
    
    # 3. Continuous soft-landing proxy: product of four bounded factors [0,1]
    #    Each factor opens gradually — provides gradient before perfect conditions.
    proximity = max(0.0, 1.0 - distance / 0.5)          # activates when dist < 0.5
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)  # activates when |v| < 0.4
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)  # activates when |angle| < 0.3
    contact   = (left_contact + right_contact) / 2.0    # 0, 0.5, or 1.0
    # Floor of 0.1 ensures weak signal before any leg contact;
    # one leg = 0.55, both legs = 1.0 — gradient toward full touchdown.
    r_landing = 8.0 * proximity * stillness * upright * (0.1 + 0.9 * contact)
    
    total_reward = r_dist + r_stability + r_landing
    
    components = {
        "distance_reward": r_dist,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing
    }
    
    return float(total_reward), components
```