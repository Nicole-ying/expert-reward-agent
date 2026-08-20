# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置 (0,0)，因为 obs 中 x,y 是相对于目标平台的坐标
    target_pos = (0.0, 0.0)

    # 1. 主学习信号：基于距离变化的进度奖励
    dist = ((obs[0] - target_pos[0]) ** 2 + (obs[1] - target_pos[1]) ** 2) ** 0.5
    next_dist = ((next_obs[0] - target_pos[0]) ** 2 + (next_obs[1] - target_pos[1]) ** 2) ** 0.5
    progress_reward = dist - next_dist  # 每一步靠近目标获得正奖励

    # 2. 稳定性惩罚（基于速度、姿态和角速度）
    vel_x = abs(next_obs[2])
    vel_y = abs(next_obs[3])
    angle = abs(next_obs[4])
    ang_vel = abs(next_obs[5])
    stability_penalty = -0.01 * (vel_x + vel_y) - 0.01 * angle - 0.01 * ang_vel

    # 3. 软着陆 proxy：在满足靠近、低速、姿态稳定且双支撑接触时给予小奖励
    near_target = (next_dist < 0.1)
    low_speed = (vel_x + vel_y < 0.2)
    stable_angle = (angle < 0.1)
    both_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_proxy = 0.5 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件与角色
- **progress_reward（主学习信号）**：通过 `(当前距离 - 下一步距离)` 给出密集的进度反馈，驱使飞行器每一步都向目标靠近。这是导航任务最核心的引导。
- **stability_penalty（稳定约束）**：轻量惩罚速度、姿态角和角速度的绝对值，鼓励飞行器平稳飞行和着陆，避免振荡、高速撞击及姿态失控。权重很小（0.01），防止抑制移动动机。
- **soft_landing_proxy（任务完成近似信号）**：当飞行器非常接近目标、低速、姿态稳定且双支撑接触时，给予一次性小奖励（0.5）。它作为“可能成功着陆”的 proxy，但**不伪造成功标志**，仅当作额外引导。

## 为什么未使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available = false`，`info` 为空字典，无法区分成功/失败终止。
- 硬编码一个基于终止状态的奖励将依赖未声明的信号，违反约束。
- 因此在 v1 中完全避免任何终点成功/失败核心项。

## 留到后续迭代的组件
- **energy_penalty / time_penalty**：待智能体学会接近并稳定后，再加入动作效率或步数代价，现阶段过早加入可能导致“不敢动”。
- **gated_reward**：目前不需要安全门控，进度奖励 + 稳定性惩罚已能初步平衡。
- **terminal_success / failure**：待环境 wrapper 显式暴露成功信号后再考虑。
- **potential_based_shaping**：如发现 progress_delta 造成目标附近震荡，可替换为势能函数。

## 训练后应关注的 failure mode
- **goal_near_oscillation**：飞行器在目标上空反复调整，distance 不再下降，但 reward 仍来自微弱的 progress 波动。
- **high_reward_without_success**：飞行器可能停在目标附近但未完全着陆，软着陆 proxy 未触发，而进度和稳定性惩罚仍给出较高总分。
- **fast_crash_near_goal**：为获取进度奖励而高速冲向目标，导致碰撞终止，此时稳定性惩罚可能来不及纠正。
- 若出现上述情况，需收紧 soft_landing 条件或调整稳定性权重。