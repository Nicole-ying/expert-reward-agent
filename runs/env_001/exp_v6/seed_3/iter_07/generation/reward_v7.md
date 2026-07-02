## 诊断

根据训练反馈分析：

1. **Agent 状态**：得分 260.64，episode 长度 314，已 target_solved。表现良好。
2. **信号结构问题**：`landing_incentive` ratio_to_progress = 236x，严重主导 total_reward。`progress_delta` 均值仅 0.002，几乎被淹没。稳定性项（angle/angvel/speed）贡献接近零。
3. **验证失败原因推测**：v6 代码的 `components` 字典包含了不在总公式中直接出现的变量（`progress_delta`、`speed_reduction`、`angle_reduction`、`angvel_reduction`、`total_reward`），而规范要求"components 只放总公式中直接出现的变量"。total_reward 公式是 `potential_reward + landing_incentive`，只有这两个是直接变量。v7 可能是系统自动生成的版本，触发了更严格的 components 校验。

**修复方案**：
- 保持 v6 的数学结构不变（已验证有效）
- 修正 components：只保留 `potential_reward` 和 `landing_incentive`
- 移除 `total_reward`（返回值本身不应出现在 components 中）
- 适当降低 `landing_incentive` 峰值（1.5→1.2），让 progress 信号有呼吸空间，同时不破坏已学会的降落行为

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- potential-based shaping: Φ = -(distance + w_speed*speed + w_angle*|angle| + w_angvel*|angvel|) ----
    # Unchanged backbone from v6 — provides the main progress gradient.

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

    # --- potential values (weights unchanged from v6) ---
    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.02

    phi_obs = -(dist_obs + w_speed * speed_obs + w_angle * angle_obs + w_angvel * angvel_obs)
    phi_next = -(dist_next + w_speed * speed_next + w_angle * angle_next + w_angvel * angvel_next)

    potential_reward = phi_next - phi_obs

    # --- landing incentive: bounded proximity gating ---
    # Peak reduced 1.5→1.2 to ease the 236x dominance over progress signal,
    # while retaining strong terminal guidance for the learned landing behavior.
    near_factor = 1.0 / (1.0 + 15.0 * dist_next)
    slow_factor = 1.0 / (1.0 + 8.0 * speed_next)
    upright_factor = 1.0 / (1.0 + 8.0 * angle_next)
    landing_incentive = 1.2 * near_factor * slow_factor * upright_factor

    # --- total ---
    total_reward = potential_reward + landing_incentive

    # --- components: only variables directly in the total_reward formula ---
    components = {
        'potential_reward': potential_reward,
        'landing_incentive': landing_incentive,
    }

    return float(total_reward), components
```