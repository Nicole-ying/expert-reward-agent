# Response Record

# Reward Revision for Env_001

## Design Changes

- **保留**: `progress_reward` 作为主学习信号，权重不变 (2.0)
- **保留**: `distance_reward` 作为持续接近引导，权重不变 (-0.1 * next_dist)
- **保留**: `landing_bonus` 作为软着陆代理，条件不变
- **削弱**: `stability_penalty` 权重进一步降低，从 `-(0.05*angle + 0.025*angular_vel + 0.1*speed)` 改为 `-(0.03*angle + 0.015*angular_vel + 0.05*speed)`，因为证据显示该惩罚项均值 (-0.110) 远大于 progress_reward 均值 (0.032)，主导了信号并导致早期崩溃
- **修改**: 无其他修改
- **新增**: 无新增组件
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**: 仍然没有显式的 success/failure 信号可用，info 字段不可靠
- **下一轮训练后应该重点观察**: stability_penalty 的绝对值是否仍然远大于 progress_reward；episode_length 是否增加；external_reward 是否提升

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta

    # 稳定约束：stability_penalty (进一步削弱)
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.03 * abs(next_obs[4])  # 从0.05降至0.03
    angular_vel_penalty = 0.015 * abs(next_obs[5])  # 从0.025降至0.015
    speed_penalty = 0.05 * speed  # 从0.1降至0.05
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)

    # 平滑接近引导：distance_reward (保留)
    distance_reward = -0.1 * next_dist

    # 任务完成proxy：soft_landing_proxy (保留)
    near_target = next_dist < 0.5
    low_speed = speed < 0.8
    stable_angle = abs(next_obs[4]) < 0.3
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_reward + stability_penalty + distance_reward + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "distance_reward": distance_reward,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```
