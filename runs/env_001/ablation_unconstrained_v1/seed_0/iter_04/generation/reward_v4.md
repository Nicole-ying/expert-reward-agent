## 分析

### 1. 这个 agent 发生了什么？

**行为证据：**
- 13/20 episode 成功终止（着陆），7/20 被截断（超时）。得分范围 [29.2, 204.5]。
- 平均 episode 长度 837 步——即便成功的 episode 也偏长。
- `distance_reward`（100% active, 47.5% share）和 `landing_proxy`（75.4% active, 52% share）占据几乎全部 reward 量级。
- `stability_penalty` 仅占 0.5% magnitude share——几乎无关紧要。

**诊断：** 当前的 `distance_reward = 1/(1+d)` 是一个**持久状态值**——只要 agent 待在目标附近就有稳定回报。`landing_proxy` 的 `(0.1 + 0.9*contact)` 底层即使双腿均未接触也有 10% 的 payout。这构成了**奖励高原**：agent 可以在目标垫附近盘旋收集 reward，而不需要真正完成着陆。65% 的 episode 最终着陆了，但 35% 在高原上徘徊直到超时。

### 2. 哪个组件最值得干预？

**`distance_reward`**（持久状态值）和 **`landing_proxy` 的 0.1 底层**。

根据 `state_to_improvement` 变换原则：将持久状态值 `q(s)` 改为改进量 `q(s') - q(s)`（progress delta），agent 只有**变得更好**时才能获得 reward，而非停留在代理最优状态。

### 3. 我之前改了什么？

上一轮设计了 bounded proximity + stability + landing proxy with contact floor 的组合。它在 65% episode 上成功着陆，但剩余 35% 证明了奖励高原的存在——距离奖励和 contact floor 共同允许盘旋套利。

**修改方案（基于 best 框架，做 4 处有证据的改动）：**

1. **距离 reward → progress delta**：`old_dist - new_dist`，只有接近目标才有正 reward，原地不动为零，远离为负。消除盘旋套利。
2. **移除 landing_proxy 的 0.1 底层**：`contact_avg` 直接使用，无接触则此分量为零。代理奖励仅在实际触腿时触发。
3. **添加 time_penalty**：`-0.003/步`，施加完成速度压力。
4. **小幅提升 landing_proxy 倍率**：`8.0 → 12.0`，补偿移除底层后的信号强度，强化真实着陆行为。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- Previous state ---
    old_x = obs[0]
    old_y = obs[1]
    old_dist = (old_x**2 + old_y**2)**0.5

    # --- Next state ---
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    new_dist = (x_pos**2 + y_pos**2)**0.5

    # 1. Progress delta — reward approaching, penalize retreating, zero when static
    r_progress = 5.0 * (old_dist - new_dist)

    # 2. Small proximity bonus — faint attractor near target (prevents gradient collapse)
    r_proximity = 0.2 / (1.0 + new_dist)

    # 3. Stability penalty — discourage violent motion
    r_stability = -(
        0.02 * (abs(x_vel) + abs(y_vel)) +
        0.15 * abs(body_angle) +
        0.08 * abs(ang_vel)
    )

    # 4. Landing proxy — only pays with leg contact (no 0.1 floor)
    proximity = max(0.0, 1.0 - new_dist / 0.5)
    stillness = max(0.0, 1.0 - (abs(x_vel) + abs(y_vel)) / 0.4)
    upright   = max(0.0, 1.0 - abs(body_angle) / 0.3)
    contact_avg = (left_contact + right_contact) / 2.0
    r_landing = 12.0 * proximity * stillness * upright * contact_avg

    # 5. Time pressure — constant gradient toward faster completion
    r_time = -0.003

    total_reward = r_progress + r_proximity + r_stability + r_landing + r_time

    components = {
        "progress_reward": r_progress,
        "proximity_bonus": r_proximity,
        "stability_penalty": r_stability,
        "landing_proxy": r_landing,
        "time_penalty": r_time
    }

    return float(total_reward), components
```