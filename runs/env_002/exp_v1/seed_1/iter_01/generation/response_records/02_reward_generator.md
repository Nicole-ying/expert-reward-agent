# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 势能函数：Phi = -|hull_angle| - 0.5 * |horizontal_velocity - target_velocity|
    # 目标水平速度设为 1.0 m/s（合理行走速度）
    target_velocity = 1.0
    
    # 当前势能
    phi_obs = -abs(obs[0]) - 0.5 * abs(obs[2] - target_velocity)
    # 下一时刻势能
    phi_next = -abs(next_obs[0]) - 0.5 * abs(next_obs[2] - target_velocity)
    
    gamma = 0.99  # 折扣因子
    progress_delta = gamma * phi_next - phi_obs
    
    # 缩放系数，使奖励在合理范围
    progress_delta_reward = 2.0 * progress_delta
    
    # ========== 稳定约束：stability_penalty ==========
    # 惩罚：主体角速度过大、垂直速度过大、主体角度过大
    angular_velocity_penalty = -0.5 * abs(next_obs[1])  # 主体角速度
    vertical_velocity_penalty = -0.3 * abs(next_obs[3])  # 垂直速度
    angle_penalty = -1.0 * abs(next_obs[0])  # 主体角度
    
    stability_penalty = angular_velocity_penalty + vertical_velocity_penalty + angle_penalty
    
    # ========== 动作代价：energy_penalty（小权重） ==========
    # 使用动作的平方和作为能量消耗的代理
    energy_penalty = -0.01 * (action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2)
    
    # ========== 总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + energy_penalty
    
    # ========== 组件字典 ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "energy_penalty": energy_penalty,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号）
   - 角色：引导智能体保持直立姿态并以目标速度前进
   - 势能函数 `Phi = -|hull_angle| - 0.5 * |horizontal_velocity - 1.0|`，同时惩罚偏离竖直方向和偏离目标行走速度
   - 使用 potential-based shaping 形式 `gamma * Phi(next_obs) - Phi(obs)`，保证策略不变性
   - 权重 2.0 使其成为主导信号

2. **stability_penalty**（稳定约束）
   - 角色：防止智能体剧烈晃动或跳跃，促进稳定步态
   - 包含三个子项：主体角速度惩罚（-0.5 * |angular_velocity|）、垂直速度惩罚（-0.3 * |vertical_velocity|）、主体角度惩罚（-1.0 * |angle|）
   - 权重适中，不会过度抑制运动

3. **energy_penalty**（动作代价）
   - 角色：鼓励节能，防止无意义的高频抖动
   - 使用动作平方和作为能量代理，权重仅 0.01，确保不会让智能体不敢运动

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`
- info 始终为空字典，无法获取成功/失败标志
- 因此 v1 不包含终端奖励，避免伪造信号

## 留到后续迭代的组件

- **terminal_success_reward**：当 wrapper 能暴露成功标志时加入
- **terminal_failure_penalty**：当能区分失败终止时加入
- **time_penalty**：如果智能体接近目标但拖太久，小权重加入
- **gated_reward**：如果安全被进度奖励抵消，再加入安全门控
- **soft_landing_proxy**：如果智能体学会前进但姿态不稳定，可加入接触条件

## 训练后应观察的 failure mode

1. **agent_afraid_to_move**：如果智能体原地不动或动作很小，说明 energy_penalty 权重过大或 progress_delta_reward 权重过小
2. **high_reward_without_success**：如果智能体获得高奖励但频繁摔倒，说明 stability_penalty 权重不足
3. **目标附近震荡**：如果智能体在目标速度附近来回摆动，说明 progress_delta_reward 的势能函数需要 clip 或调整
4. **步态不自然**：如果智能体用奇怪姿势前进但奖励高，说明需要加入步态周期或对称性约束
