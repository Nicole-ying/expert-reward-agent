好的，这是根据上一轮奖励函数和迭代上下文进行的修订。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测变量
    x_pos = obs[0]
    y_pos = obs[1]
    x_vel = obs[2]
    y_vel = obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact = obs[6]
    right_contact = obs[7]
    
    next_x_pos = next_obs[0]
    next_y_pos = next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]
    
    # 1. 主学习信号：progress_delta_reward
    # 计算当前位置到目标(0,0)的距离
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 10.0 * progress_delta
    
    # 2. 稳定/安全约束：stability_penalty (削弱)
    # 惩罚高速、大姿态角和大角速度，但降低权重以避免主导progress信号
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    angle_penalty = 0.2 * abs(next_body_angle)
    angular_vel_penalty = 0.1 * abs(next_ang_vel)
    speed_penalty = 0.1 * speed
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 3. 任务完成proxy：soft_landing_proxy (保留，但条件更宽松)
    # 当接近目标、低速、姿态稳定且双接触时给予小奖励
    near_target = next_dist < 0.8
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.3
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # 4. 动作代价：energy_penalty (删除)
    # 上一轮中该惩罚项贡献很小，且可能干扰早期探索，故移除
    
    # 5. 新增：distance_reward (小锚点)
    # 当progress信号不足时，提供持续向目标靠近的微弱鼓励
    distance_reward = -0.1 * next_dist
    
    # 组合总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus + distance_reward
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "distance_reward": distance_reward,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

### 设计说明

- **保留**：`progress_delta_reward` 作为主要学习信号，其非零率接近100%，证明其有效。
- **削弱**：`stability_penalty` 的权重被降低（角度惩罚从0.5降至0.2，角速度从0.2降至0.1，速度从0.3降至0.1）。上一轮证据显示其绝对值均值（0.343）是progress信号（0.160）的两倍多，很可能主导了梯度并导致智能体过早崩溃。降低权重以平衡探索与稳定。
- **删除**：`energy_penalty`。其贡献很小（-0.1 * engine_use），且可能不必要地惩罚早期探索动作。移除它简化了奖励结构。
- **修改**：`landing_bonus` 的条件被放宽（距离从0.5放宽到0.8，速度从0.3放宽到0.5，角度从0.2放宽到0.3），使其更容易被触发，从而提供更平滑的着陆引导，而不是一个稀疏的硬奖励。
- **新增**：`distance_reward`，一个小的负距离项（-0.1 * next_dist）。这为智能体提供了一个持续的、微弱的“向目标靠近”的锚点信号，特别是在progress信号很小时（例如，当智能体在目标附近振荡时）。这有助于解决“early_failure_or_crash”问题，因为它鼓励智能体不要远离目标。
- **为什么仍然不使用 terminal_success_reward / terminal_failure_penalty**：因为 `info` 字典中仍然没有可靠的 `success` 或 `failure` 信号。使用这些奖励会引入不正确的学习信号。
- **下一轮训练后应该重点观察**：
    1.  `progress_reward` 和 `stability_penalty` 的均值比例是否更平衡（例如，progress的绝对值应大于或等于stability的绝对值）。
    2.  `landing_bonus` 的触发率是否提高（nonzero_rate）。
    3.  `distance_reward` 的均值是否稳定为负且绝对值较小，表明智能体在持续靠近目标。
    4.  外部评估奖励（external evaluation reward）是否有所提升，以及平均回合长度是否增加。