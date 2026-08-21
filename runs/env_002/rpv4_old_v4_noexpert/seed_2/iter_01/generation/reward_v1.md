```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Locomotion reward: forward progress + upright stability.
    # Only uses environment_card-declared obs/next_obs indices.

    # --- forward progress (mandatory role) ---
    forward_weight = 1.0
    speed_scale = 1.0
    forward_speed = 0.5 * (obs[2] + next_obs[2])
    forward_progress_reward = (
        forward_weight
        * forward_speed
        / ((forward_speed ** 2 + speed_scale ** 2) ** 0.5)
    )

    # --- upright stability (mandatory health constraint) ---
    angle_deadband = 0.5
    angvel_deadband = 1.5
    angle_weight = 1.0
    angvel_weight = 0.05
    eps = 1e-6

    avg_angle = 0.5 * (obs[0] + next_obs[0])
    avg_angvel = 0.5 * (obs[1] + next_obs[1])

    angle_raw = abs(avg_angle) - angle_deadband
    angle_excess = 0.5 * (angle_raw + (angle_raw ** 2 + eps) ** 0.5)
    angle_penalty = angle_weight * angle_excess ** 2

    angvel_raw = abs(avg_angvel) - angvel_deadband
    angvel_excess = 0.5 * (angvel_raw + (angvel_raw ** 2 + eps) ** 0.5)
    angvel_penalty = angvel_weight * angvel_excess ** 2

    upright_stability_penalty = -(angle_penalty + angvel_penalty)

    total_reward = forward_progress_reward + upright_stability_penalty

    components = {
        "forward_progress_reward": float(forward_progress_reward),
        "upright_stability_penalty": float(upright_stability_penalty)
    }

    return float(total_reward), components
```

## 设计说明

本环境是 2D 双足连续控制前进任务，核心目标是稳定持续前进。由于没有绝对位置、终点距离、success flag 或 failure flag，v1 奖励只依赖可观测的躯干角度、角速度和水平速度信号。

### 已选 reward roles

1. **forward_progress（主学习信号）**  
   - 使用 `obs[2]` 和 `next_obs[2]` 的平均水平速度。  
   - 采用有界归一化形式 `x / sqrt(x^2 + scale^2)`，避免速度较大时线性奖励无界膨胀，同时对小速度保持梯度。  
   - 前进为正奖励，后退为负奖励，鼓励持续向前。

2. **upright_stability（健康/安全约束）**  
   - 使用 `obs[0]`、`next_obs[0]` 的平均躯干倾角，以及 `obs[1]`、`next_obs[1]` 的平均躯干角速度。  
   - 对角度设置 0.5 rad 死区，对角速度设置 1.5 rad/s 死区。  
   - 死区外采用二次惩罚，使用软 ReLU 形式保持平滑。该约束用于降低摔倒概率，但避免对正常步态中的轻微姿态波动过度惩罚。

### 未加入 v1 的职责

- `energy_efficiency_penalty`：环境明确建议早期不要强加，否则可能让智能体选择不动或极小动作。  
- `terrain_aware_lidar_penalty`：LIDAR 信号初期不确定，容易引入虚假惩罚。  
- `terminal_success_reward` / `terminal_failure_penalty`：环境中没有显式 success/failure flag，info 为空，不能使用。  
- `goal_distance_reward`：没有绝对位置或到终点距离，不可用。  
- `original_reward_proxy`：官方原始奖励被禁止使用。