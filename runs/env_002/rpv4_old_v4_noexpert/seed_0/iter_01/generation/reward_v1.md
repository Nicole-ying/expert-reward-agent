## 设计说明

本版本仅依据 environment_card 设计，没有使用 expert schema 模板或 formula operator library。

**采用 roles**
- `forward_progress`（mandatory）：使用 `next_obs[2]` 水平速度。对正向速度给线性奖励，并限制在 `[0, 2]` 内；对后退速度给略强的负奖励，避免前后摆动但平均前进很低。
- `torso_stability`（mandatory）：使用 `next_obs[0]` 躯干倾角和 `next_obs[1]` 躯干角速度，做二次惩罚，抑制摔倒。
- `vertical_velocity_penalty`（conditional health/anti-bounce）：使用 `next_obs[3]` 垂直速度，施加较小二次惩罚，避免利用躯干弹跳而不是腿部前进。

**未使用**
- `energy_regularization`：v1 先学会走路，避免早期压制有效动作。
- `contact_patterning`：崎岖地形上强制接触模式可能误导。
- `sparse_goal_reward` / `official_reward_matching` / terminal success/failure：环境中均不可用或被禁止。
- `lidar_obstacle_penalty`：lidar 语义不明确，v1 不使用。
- `training_progress`：本任务未允许使用。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    horizontal_velocity = next_obs[2]

    # Forward progress: encourage positive horizontal velocity, penalize backward motion.
    if horizontal_velocity >= 0.0:
        forward_progress = 1.0 * min(horizontal_velocity, 2.0)
    else:
        forward_progress = 1.5 * max(horizontal_velocity, -2.0)

    hull_angle = next_obs[0]
    hull_angular_velocity = next_obs[1]
    vertical_velocity = next_obs[3]

    torso_stability = -0.15 * (hull_angle ** 2) - 0.05 * (hull_angular_velocity ** 2)
    vertical_velocity_penalty = -0.08 * (vertical_velocity ** 2)

    total_reward = forward_progress + torso_stability + vertical_velocity_penalty

    components = {
        "forward_progress": forward_progress,
        "torso_stability": torso_stability,
        "vertical_velocity_penalty": vertical_velocity_penalty,
    }

    return float(total_reward), components
```