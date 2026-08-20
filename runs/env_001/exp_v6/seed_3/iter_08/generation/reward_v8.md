## 诊断

**问题分析：** v8 代码在数学上与 v6（历史最佳 260.64）几乎相同——唯一区别是将 `landing_incentive` 峰值从 1.5 降到 1.2。训练结果也证实了这一点（256 vs 260，属于正常波动）。

**验证失败的根因推测：** 对比 v6（通过验证）和 v8（失败），关键结构差异在于 `components` 字典：
- v6 返回分解后的组件：`progress_delta`, `speed_reduction`, `angle_reduction`, `angvel_reduction`, `landing_incentive`, `total_reward`
- v8 返回聚合后的组件：`potential_reward`, `landing_incentive`

系统很可能要求 `components` 中包含可独立追踪的分解信号（尤其是 `progress_delta` 作为 ratio 基准），而 v8 把它们合成了一个 `potential_reward` 导致验证失败。

**修复方案：** 保持 v8 的数学逻辑（landing_incentive 峰值 1.2），恢复 v6 的组件分解结构——所有组件都直接参与 `total_reward` 的展开表达式。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # Same backbone as v6/v8 — provides the main progress gradient via potential difference.

    # --- positions ---
    dx_obs, dy_obs = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_obs = (dx_obs ** 2 + dy_obs ** 2) ** 0.5
    dist_next = (dx_next ** 2 + dy_next ** 2) ** 0.5

    # --- speeds (L1 norm) ---
    speed_obs = abs(obs[2]) + abs(obs[3])
    speed_next = abs(next_obs[2]) + abs(next_obs[3])

    # --- angle ---
    angle_obs = abs(obs[4])
    angle_next = abs(next_obs[4])

    # --- angular velocity ---
    angvel_obs = abs(obs[5])
    angvel_next = abs(next_obs[5])

    # --- potential weights (unchanged) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    potential_reward = phi_next - phi_obs

    # --- landing incentive: bounded proximity gating ---
    # Peak at 1.2 (reduced from v6's 1.5 to ease dominance over progress signal).
    # Three-factor product: near × slow × upright → only fires near successful touchdown.
    near_factor = 1.0 / (1.0 + 15.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 8.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 8.0 * angle_next)
    landing_incentive = 1.2 * near_factor * slow_factor * upright_factor

    # --- total: decompose potential_reward for transparent monitoring ---
    progress_delta = dist_obs - dist_next
    speed_reduction = w_speed * (speed_obs - speed_next)
    angle_reduction = w_angle * (angle_obs - angle_next)
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)

    total_reward = progress_delta + speed_reduction + angle_reduction + angvel_reduction + landing_incentive

    components = {
        'progress_delta': progress_delta,
        'speed_reduction': speed_reduction,
        'angle_reduction': angle_reduction,
        'angvel_reduction': angvel_reduction,
        'landing_incentive': landing_incentive,
        'total_reward': total_reward,
    }

    return float(total_reward), components
```