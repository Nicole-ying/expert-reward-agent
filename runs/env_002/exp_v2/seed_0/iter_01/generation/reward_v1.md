# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 使用水平速度作为前进进度的代理
    # 奖励每一步水平速度的增加（即加速度为正）
    current_horizontal_vel = obs[2]
    next_horizontal_vel = next_obs[2]
    progress_delta = next_horizontal_vel - current_horizontal_vel
    progress_delta_reward = 2.0 * progress_delta  # 权重2.0，鼓励持续加速

    # ========== 稳定约束：stability_penalty ==========
    # 惩罚身体倾斜角度过大和角速度过大，防止摔倒
    hull_angle = next_obs[0]  # 身体倾斜角度
    hull_angular_vel = next_obs[1]  # 身体角速度
    # 使用平滑的指数惩罚，温度参数控制惩罚的陡峭程度
    temperature_angle = 0.5
    temperature_angular_vel = 0.3
    angle_penalty = -0.5 * (2.718281828 ** (abs(hull_angle) / temperature_angle) - 1.0)
    angular_vel_penalty = -0.3 * (2.718281828 ** (abs(hull_angular_vel) / temperature_angular_vel) - 1.0)
    stability_penalty = angle_penalty + angular_vel_penalty

    # ========== 效率约束：energy_penalty（小权重） ==========
    # 惩罚过大的关节力矩，鼓励节能
    action_magnitude = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = -0.01 * action_magnitude  # 权重很小，避免agent不敢动

    # ========== 任务完成proxy：soft_landing_proxy ==========
    # 当身体直立、速度适中、双脚着地时给予小奖励
    # 注意：不能直接使用contact作为success，必须组合条件
    leg1_contact = next_obs[12]
    leg2_contact = next_obs[13]
    both_contact = 1.0 if (leg1_contact > 0.5 and leg2_contact > 0.5) else 0.0
    # 直立条件：角度小于0.2弧度，角速度小于0.5
    is_upright = 1.0 if (abs(hull_angle) < 0.2 and abs(hull_angular_vel) < 0.5) else 0.0
    # 速度适中：水平速度在0.5到3.0之间
    is_good_speed = 1.0 if (0.5 < next_horizontal_vel < 3.0) else 0.0
    soft_landing_bonus = 0.5 * both_contact * is_upright * is_good_speed

    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty + soft_landing_bonus

    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号）
   - 角色：核心学习引导，奖励每一步水平速度的增加（即加速度为正）
   - 使用 `obs[2]` 和 `next_obs[2]` 计算速度增量
   - 权重2.0，鼓励智能体持续加速前进

2. **stability_penalty**（稳定/安全约束）
   - 角色：防止身体倾斜过大或角速度过快导致摔倒
   - 使用指数形式的平滑惩罚，避免硬边界导致梯度消失
   - 两个子项：角度惩罚（温度0.5）和角速度惩罚（温度0.3）

3. **energy_penalty**（效率/动作代价）
   - 角色：小权重约束，防止无意义的剧烈动作
   - 使用动作力矩的平方和，权重仅0.01，避免agent不敢动

4. **soft_landing_bonus**（任务完成proxy）
   - 角色：当智能体同时满足直立、速度适中、双脚着地时给予小奖励
   - 组合条件：`both_contact * is_upright * is_good_speed`
   - 权重0.5，作为辅助信号引导稳定步态

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info字典始终为空，无法获取成功或失败的显式信号
- 终止条件（摔倒、到达终点）发生在step之后，无法在终止时获取奖励信号
- 因此v1不使用终端奖励，而是通过连续信号引导学习

## 留到后续迭代的组件

- **terminal_success_reward**：当wrapper能暴露成功信号时再加入
- **terminal_failure_penalty**：当能区分失败终止原因时再加入
- **time_penalty**：如果智能体拖太久不前进，后续可小权重加入
- **gated_reward**：如果安全约束被进度奖励抵消，后续可加入门控
- **potential_based_shaping**：如果需要更标准的塑形，后续可替换progress_delta

## 训练后应观察的failure mode

1. **agent_afraid_to_move**：如果energy_penalty权重过大，智能体可能不敢动
2. **high_reward_without_success**：如果progress_delta_reward主导，智能体可能原地抖动获得高奖励而不前进
3. **unstable_gait**：如果stability_penalty过弱，智能体可能采用不稳定但速度快的步态
4. **contact_reward_hacking**：如果soft_landing_bonus条件太宽，智能体可能通过频繁触地获得奖励而不前进