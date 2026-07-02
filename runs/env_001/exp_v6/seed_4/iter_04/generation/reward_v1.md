# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 势能塑形：引导靠近目标点
    gamma = 0.99
    # 采用负欧几里得距离作为势能函数
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    shaping_reward = gamma * (-dist_next) - (-dist_current)  # = dist_current - gamma * dist_next

    # 2. 稳定性惩罚：基于下一时刻的速度、姿态、角速度
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    w_vel = 0.05        # 速度项系数
    w_ang = 0.02        # 角速度项系数
    w_angle = 0.05      # 姿态角系数

    stability_penalty = -(
        w_vel * (abs(vel_x) + abs(vel_y)) +
        w_ang * abs(ang_vel) +
        w_angle * abs(angle)
    )

    total_reward = shaping_reward + stability_penalty

    components = {
        "potential_shaping_reward": shaping_reward,
        "stability_penalty": stability_penalty,
        "total_reward": total_reward
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色
1. **potential_shaping_reward（主学习信号，势能塑形）**  
   - **角色**：密集引导飞行器向目标垫（obs[0]≈0, obs[1]≈0）移动。  
   - 此组件可代替已被尝试的 `progress_delta_reward` 系列，通过带有折扣因子 γ 的势能差分使每一步接近行为获得正奖励。采用负距离作为势能函数，该设计从根本上区别于直接的位置增量差分，是新的学习骨架。  
   - 权重隐含为 1.0，无需额外缩放。

2. **stability_penalty（稳定约束）**  
   - **角色**：惩罚不稳定的高速移动、剧烈旋转和姿态倾斜，促使最终接近时速度收敛、姿态垂直，为安全着陆提供前置条件。  
   - 基于下一状态的速度、角速度和机体倾斜角给出温和的惩罚项，权重较小（速度系数 0.05，角速度 0.02，角度 0.05），避免过度抑制探索。

## 未使用 terminal_success_reward / terminal_failure_penalty 的原因
- 根据环境卡片，`explicit_success_flag_available = false` 且 `explicit_failure_flag_available = false`，info 字典为空，**不存在任何可用的成功/失败信号**。  
- 强行加入此类终点奖励可能诱导函数错误地发明未提供的标志位，或对终止原因做出不可靠假设，违背安全原则。

## 暂不加入并留至后续迭代的组件
- **soft_landing_proxy**：当前版本侧重基础靠近与稳定；待 `potential_shaping` 让智能体能稳定着陆区域后，再根据接触信号和低速条件加入小权重“任务完成近似奖励”，否则可能导致接触奖励黑入。  
- **terminal_success_reward / terminal_failure_penalty**：一旦环境提供了可靠的终止状态标识，再引入以获得更强的目标信号。  
- **energy_penalty / time_penalty**：当前阶段不宜过早惩罚动作或步数，以免阻碍探索；后续若出现高燃料消耗或长时间悬停再添加。

## 训练后应重点观察的 failure mode
- **high_reward_without_success**：由于势能塑形可能在靠近目标后因距离变化很小而持续获得接近零的正奖励，但未真正完成着陆（未接触平台或姿态错误），需检查成功率是否同步提升。  
- **goal_near_oscillation**：势能塑形配合稳定性惩罚可能导致在目标附近来回震荡而无法稳定着陆；若出现应适当调整稳定性惩罚的权重或加入更严格的着陆条件。  
- **fast_crash_near_goal**：虽然有稳定性惩罚，但若权重过轻，可能仍出现高速撞击着陆区；必要时可增大速度惩罚或在后继版本中引入失败惩罚。  
- **agent_afraid_to_move**：如果稳定性惩罚过强，智能体可能选择完全不动以获取稳定奖励，需监控平均速度是否接近零并伴随低进度。