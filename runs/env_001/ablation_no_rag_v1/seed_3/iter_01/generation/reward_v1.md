# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next state variables
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to landing pad center (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # 1. Main learning signal: progress toward the pad
    progress_reward = dist - next_dist

    # 2. Stability penalty: penalize high velocities
    w_vel = 0.1
    velocity_penalty = -w_vel * (vx**2 + vy**2)

    # 3. Attitude penalty: penalize non-zero body angle (want upright)
    w_angle = 0.01
    attitude_penalty = -w_angle * abs(angle)

    # 4. Soft landing proxy: encourage gentle touchdown with both legs
    w_landing = 2.0
    alpha = 20.0   # sharpness for position proximity
    beta = 5.0     # sharpness for low speed
    contact_factor = left_contact * right_contact  # 1 if both legs touch, else 0
    position_proximity = 2.718281828 ** (-alpha * (next_x**2 + next_y**2))
    speed_term = 2.718281828 ** (-beta * (vx**2 + vy**2))
    landing_proxy = w_landing * contact_factor * position_proximity * speed_term

    total_reward = progress_reward + velocity_penalty + attitude_penalty + landing_proxy

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "attitude_penalty": attitude_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 奖励组件与角色
- **progress_reward**（主学习信号）：使用当前时刻到下一时刻的欧氏距离减少量作为正向奖励。该信号稠密、每步可导，直接驱动主体向目标（着陆垫中心）移动，是所有任务行为的核心驱动力。
- **velocity_penalty**（稳定约束）：对平移速度大小的二次惩罚，权重较小。它鼓励主体在接近目标时减速，避免高速坠毁，同时不影响早期的快速接近（因为 progress_reward 主导方向）。
- **attitude_penalty**（姿态平滑约束）：惩罚偏离竖直的姿态角，权重很小。有助于保持机身稳定，防止着陆时侧翻，但不强制完美姿态。
- **landing_proxy**（任务完成近似信号）：使用双腿触地、位置近中心、速度低的三个条件连续乘积。乘积形式使奖励只有在三个条件都趋向满足时才显著，提供稀疏但关键的“最终着陆”信息。它避免了单一二值触地的奖励作弊风险。

## 为什么没有 terminal_success_reward / terminal_failure_penalty
环境未提供显式的成功或失败标志（`info` 为空，`explicit_success_flag_available=false`），依据原则禁止使用不存在的字段。我们通过稠密奖励和 soft proxy 间接引导成功行为，不依赖硬编码的终端事件。

## 留到后续迭代的组件
- 引擎使用效率惩罚（例如 action==2 惩罚）—— 当前版本优先让智能体学会完成任务，重量级动作代价将在后续版本加入。
- 更精细的触地时序奖励或阶段门控 —— V1 已有一个简单的 soft proxy，未来可基于学习曲线替换为解耦的 multi-stage curriculum。
- 速度朝向修正（如只在速度背离目标时惩罚）—— 当前全向速度惩罚可能对早期探索有轻微抑制，后续可改进为方向性惩罚以释放更多“冲刺”自由。

## 训练后应观察的 failure mode
1. **在目标附近高速震荡**：如果 velocity_penalty 权重太小或 landing_proxy 不够锐利，智能体可能为了赚取 progress_reward 而反复穿越目标区而不减速着陆。
2. **过早趴窝**：如果 attitude_penalty 或 velocity_penalty 过强，智能体可能选择原地不动来避免惩罚，忽略 progress_reward。
3. **双腿未同时触地的假成功**：landing_proxy 的双腿接触条件要求两腿同时为 1。如果环境允许仅单腿触垫而终止，奖励可能不足，导致无法稳定着陆。
4. **只靠主发动机悬停而不着垫**：landing_proxy 的稀疏特性可能在训练早期被忽略，需要观察是否收敛到最终着陆行为。若长时间未发生触地，可能需要提高 w_landing 或调整 alpha/beta。