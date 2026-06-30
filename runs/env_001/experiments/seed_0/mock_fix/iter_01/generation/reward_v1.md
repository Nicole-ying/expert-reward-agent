# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    old_x = obs[0]
    old_y = obs[1]
    new_x = next_obs[0]
    new_y = next_obs[1]
    new_vx = next_obs[2]
    new_vy = next_obs[3]
    new_angle = next_obs[4]
    new_angular_velocity = next_obs[5]

    old_distance = (old_x * old_x + old_y * old_y) ** 0.5
    new_distance = (new_x * new_x + new_y * new_y) ** 0.5
    next_speed = (new_vx * new_vx + new_vy * new_vy) ** 0.5

    progress_reward = old_distance - new_distance
    stability_penalty = -0.03 * next_speed - 0.03 * abs(new_angle) - 0.01 * abs(new_angular_velocity)

    total_reward = progress_reward + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    if isinstance(info, dict):
        info["reward_terms"] = components

    return float(total_reward), components
```

# reward_v1 设计说明

- 使用组件：progress_delta_reward + light stability_penalty。
- progress_delta_reward 负责主学习信号：奖励智能体每一步更接近目标区域。
- stability_penalty 负责轻量稳定约束：防止高速、姿态过大或角速度过大。
- 没有使用 terminal_success_reward，因为环境卡片说明 explicit_success_flag_available=false。
- 没有使用 terminal_failure_penalty，因为没有 explicit_failure_flag 或 termination_reason。
- 后续迭代可以根据训练现象加入 derived soft landing proxy、energy_penalty、time_penalty 或 gated_reward。
- 训练后重点观察：goal_near_oscillation、fast_crash_near_goal、high_reward_without_success、agent_afraid_to_move。
