# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress_delta_reward
    # 计算当前位置到目标（0,0）的距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = current_dist - next_dist
    progress_reward = 2.0 * progress_delta  # 权重2.0，鼓励每一步更接近目标
    
    # 稳定约束：stability_penalty
    # 惩罚速度、姿态角和角速度，鼓励稳定接近
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_penalty = 0.1 * abs(next_obs[4])  # 姿态角惩罚，权重0.1
    angular_vel_penalty = 0.05 * abs(next_obs[5])  # 角速度惩罚，权重0.05
    speed_penalty = 0.2 * speed  # 速度惩罚，权重0.2
    stability_penalty = -(angle_penalty + angular_vel_penalty + speed_penalty)
    
    # 任务完成proxy：soft_landing_proxy
    # 当接近目标、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = next_dist < 0.3
    low_speed = speed < 0.5
    stable_angle = abs(next_obs[4]) < 0.2
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    landing_bonus = 1.0 if (near_target and low_speed and stable_angle and both_contact) else 0.0
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + landing_bonus
    
    # 组件记录
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "landing_bonus": landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号，权重2.0）
   - 角色：每一步更接近目标时给予正奖励，远离时给予负奖励，提供密集的导航引导。
   - 数学形式：`current_dist - next_dist`，其中距离为到原点(0,0)的欧氏距离。

2. **stability_penalty**（稳定约束，负值）
   - 角色：惩罚高速、大姿态角和快速旋转，鼓励飞行器以稳定姿态接近目标，避免撞击或失控。
   - 包含三个子项：速度惩罚（权重0.2）、姿态角惩罚（权重0.1）、角速度惩罚（权重0.05）。

3. **soft_landing_proxy**（任务完成近似信号，权重1.0）
   - 角色：当飞行器同时满足接近目标（距离<0.3）、低速（速度<0.5）、姿态稳定（角度<0.2）且双支撑接触时，给予1.0的额外奖励。
   - 这是对成功着陆的软近似，不依赖显式success flag。

## 为什么没有使用terminal_success_reward / terminal_failure_penalty

- 环境卡片明确标注`explicit_success_flag_available=false`和`explicit_failure_flag_available=false`，无法可靠区分成功和失败终止。
- 使用这些项会引入对终止原因的猜测，可能导致错误的学习信号。

## 留到后续迭代的组件

- **energy_penalty**：当前动作是离散的，且过早加入能耗惩罚可能导致agent不敢移动。后续当agent能稳定到达目标后再加入。
- **time_penalty**：当前任务没有时间限制，过早加入可能导致agent冒险。后续如果观察到agent在目标附近徘徊不完成，再加入小权重时间惩罚。
- **gated_reward**：当前安全约束通过stability_penalty实现，不需要复杂门控。后续如果出现危险行为被奖励抵消的情况，再考虑加入。

## 训练后应观察的failure mode

1. **goal_near_oscillation**：agent在目标附近来回移动，无法稳定着陆。表现为progress_reward在0附近震荡，但landing_bonus始终为0。
2. **high_reward_without_success**：agent获得高progress_reward但从未触发landing_bonus，说明它学会了接近目标但不会完成着陆。
3. **fast_crash_near_goal**：agent高速冲向目标然后坠毁。表现为高progress_reward但突然终止，stability_penalty在终止前很大。
4. **agent_afraid_to_move**：stability_penalty权重过大导致agent不敢移动。表现为所有奖励接近0，agent停留在起始位置。