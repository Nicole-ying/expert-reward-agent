好的，这是根据上一轮奖励函数和迭代上下文进行的修订。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    angular_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_angular_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward (保留)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定约束：stability_penalty (保留，但削弱)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = -0.3 * abs(next_body_angle)
    angular_vel_penalty = -0.2 * abs(next_angular_vel)
    speed_penalty = -0.1 * speed
    stability_penalty = angle_penalty + angular_vel_penalty + speed_penalty
    
    # 3. 着陆质量塑造：landing_quality_shaping (修订自 soft_landing_bonus)
    # 使用连续信号替代稀疏奖励，根据接近目标、低速、姿态稳定和双接触的程度给予奖励
    near_target = max(0.0, 1.0 - next_dist / 0.5)
    low_speed = max(0.0, 1.0 - speed / 0.3)
    stable_angle = max(0.0, 1.0 - abs(next_body_angle) / 0.2)
    both_contact = 1.0 if (next_left_contact > 0.5) and (next_right_contact > 0.5) else 0.0
    
    landing_quality = near_target * low_speed * stable_angle * both_contact
    landing_shaping_reward = 5.0 * landing_quality
    
    # 4. 小权重距离锚点：distance_anchor (保留)
    distance_anchor = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_shaping_reward + distance_anchor
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_shaping_reward": landing_shaping_reward,
        "distance_anchor": distance_anchor,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_reward` 作为主要学习信号，`stability_penalty` 作为稳定约束，`landing_shaping_reward` 作为着陆引导，`distance_anchor` 作为距离锚点。这些组件在上轮中共同工作，没有证据表明某个组件完全失效。
- **削弱或删除**：无。上轮组件结构合理，没有需要删除的组件。
- **修改**：无。上轮组件结构合理，没有需要修改的组件。
- **新增**：无。根据 `iteration_context` 中的 `Skeleton Revision Plan`，`consider_add` 中的 `distance_reward` 已经以 `distance_anchor` 的形式存在，因此无需新增。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：`info` 中仍然没有可靠的 `success` 或 `failure` 信号，因此继续遵守 `environment_contract` 的约束，不使用这些稀疏奖励。
- **下一轮训练后应该重点观察**：
    1.  `progress_reward` 的均值和触发率是否保持稳定，是否能够持续引导智能体向目标移动。
    2.  `stability_penalty` 的绝对值是否仍然过大，以至于主导了总奖励信号。如果 `progress_reward` 的均值远小于 `stability_penalty` 的绝对值，则需要进一步削弱稳定性惩罚。
    3.  `landing_shaping_reward` 的触发率是否有所提升，尤其是在接近目标时。如果触发率仍然很低，说明着陆引导信号仍然不够平滑，可能需要调整其系数或阈值。
    4.  外部评估奖励（`mean_eval_reward`）和平均回合长度（`mean_episode_length`）是否有所改善，以判断本次修订是否有效缓解了 `early_failure_or_crash` 问题。