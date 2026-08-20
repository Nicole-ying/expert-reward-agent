# Response Record

## 诊断

**1. Agent 状态：** 得分从 -100 跃升至 +84.8，episode length 823（偏长）。progress_delta 是唯一有效信号（mean=0.004），speed/angle/angvel 三项贡献接近零甚至微负。Agent 可能在目标附近徘徊但未高效降落——匹配 `goal_near_oscillation` 模式。

**2. 根因：** 势能塑形对各维度独立给 reward，但没有「同时满足」的协同信号。Agent 可以靠近但速度快、速度慢但离得远——分别得分，不形成合力引导降落。

**3. 上次改动：** 从 penalty 体系完全重建为 potential-based shaping（层次 3），得分暴涨。现在需要层次 1/2 的微调——添加连续乘积 landing incentive，解决「靠近但不降落」问题。

## 修改方案

保持势能塑形不变，新增一个 **连续乘积 landing_incentive**：`near_factor × slow_factor × upright_factor`，三个 bounded 因子相乘，只在同时接近、低速、直立时给出显著奖励。权重 0.5，远处可忽略不计，近处逐步增大。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # reward = Φ(next_obs) - Φ(obs) = sum of improvements across all dimensions.
    # This is unchanged from the successful previous iteration.

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

    # --- potential values (unchanged weights) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    # --- reward from potential improvement ---
    potential_reward = phi_next - phi_obs

    # --- landing incentive: continuous multiplicative signal ---
    # Only gives meaningful reward when near AND slow AND upright simultaneously.
    # Uses bounded 1/(1+kx) form on next_obs so it provides gradient toward good landing states.
    near_factor = 1.0 / (1.0 + 3.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 2.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 3.0 * angle_next)
    landing_incentive = 0.5 * near_factor * slow_factor * upright_factor

    # --- total ---
    total_reward = potential_reward + landing_incentive

    # --- decompose for logging ---
    progress_delta = dist_obs - dist_next
    speed_reduction = w_speed * (speed_obs - speed_next)
    angle_reduction = w_angle * (angle_obs - angle_next)
    angvel_reduction = w_angvel * (angvel_obs - angvel_next)

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
