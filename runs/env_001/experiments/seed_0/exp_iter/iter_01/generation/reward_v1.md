# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations
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
    
    # --- Component 1: Progress Delta Reward (main learning signal) ---
    # Distance to target (target is at origin: x=0, y=0)
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    progress_delta = current_dist - next_dist  # positive if moving closer
    progress_reward = 10.0 * progress_delta  # scale to meaningful magnitude
    
    # --- Component 2: Stability Penalty (lightweight constraint) ---
    # Penalize high speed, large angle, and high angular velocity
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5
    angle_penalty = 0.5 * (body_angle ** 2)  # quadratic penalty for angle deviation
    speed_penalty = 0.1 * speed  # linear penalty for speed
    angular_penalty = 0.2 * (angular_vel ** 2)  # quadratic penalty for angular velocity
    stability_penalty = -(angle_penalty + speed_penalty + angular_penalty)
    
    # --- Component 3: Soft Landing Proxy (small bonus for good landing conditions) ---
    # Conditions: near target, low speed, stable angle, both supports in contact
    near_target = next_dist < 0.5
    low_speed = speed < 0.5
    stable_angle = abs(next_body_angle) < 0.2
    both_contact = (next_left_contact > 0.5) and (next_right_contact > 0.5)
    
    soft_landing_bonus = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_bonus = 2.0  # small bonus for achieving landing conditions
    
    # --- Total Reward ---
    total_reward = progress_reward + stability_penalty + soft_landing_bonus
    
    # --- Components dict ---
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **Progress Delta Reward** (主学习信号，权重 10.0)
   - 角色：提供密集的、基于距离变化的过程引导，鼓励智能体每一步都更接近目标。
   - 数学形式：`10.0 * (current_dist - next_dist)`，其中距离为欧几里得距离。
   - 选择理由：这是 navigation_goal_reaching 任务最直接、最有效的主信号，能引导智能体从起点向目标移动。

2. **Stability Penalty** (稳定/安全约束)
   - 角色：轻量级约束，惩罚大姿态角、高速度和高角速度，防止智能体在接近目标时失控或姿态不稳。
   - 数学形式：`-(0.5 * angle^2 + 0.1 * speed + 0.2 * angular_vel^2)`。
   - 选择理由：任务要求"稳定在着陆台"，因此需要早期引入姿态和速度约束，避免智能体只关注接近目标而忽略稳定性。

3. **Soft Landing Proxy** (任务完成近似信号，小权重 2.0)
   - 角色：当智能体同时满足接近目标、低速、稳定姿态和双支撑接触时，给予小奖励，作为成功着陆的软代理。
   - 数学形式：条件触发式小奖励，非持续信号。
   - 选择理由：由于没有显式 success flag，需要一种方式鼓励智能体完成最终着陆动作，但权重很小以避免 reward hacking。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确说明 `explicit_success_flag_available=false` 和 `explicit_failure_flag_available=false`，info 字典为空。
- 使用 terminal_success_reward 会诱导发明 info['success']，违反设计原则。
- 使用 terminal_failure_penalty 需要区分失败终止原因，但当前无法可靠判断。

## 留到后续迭代的组件

- **Energy/Action Penalty**：v1 未加入，因为过早加入可能导致智能体不敢使用引擎，无法学习基本移动。后续当智能体能稳定到达目标后再优化燃料消耗。
- **Time Penalty**：v1 未加入，因为可能鼓励冒险行为。后续如果发现智能体在目标附近徘徊太久再考虑加入。
- **Gated Reward**：v1 未使用，因为门控机制可能过于复杂，且当前稳定性惩罚已提供基本安全约束。
- **Terminal Rewards**：等待 wrapper 明确暴露 success/failure 信号后再加入。

## 训练后应观察的 failure mode

1. **目标附近震荡 (goal_near_oscillation)**：如果 progress_delta_reward 导致智能体在目标附近来回移动而不稳定着陆，需要调整稳定性惩罚权重或收紧 soft_landing_proxy 条件。
2. **高速撞击 (fast_crash_near_goal)**：如果智能体以高速接近目标并坠毁，说明速度惩罚不足，需要增大 speed_penalty 权重。
3. **姿态失稳 (unstable_approach)**：如果智能体以倾斜姿态接近目标，需要增大 angle_penalty 或 angular_penalty。
4. **接触奖励滥用 (contact reward hacking)**：如果智能体通过非正常方式触发接触（如侧翻），需要收紧 soft_landing_proxy 条件或移除该组件。
5. **不敢移动 (agent_afraid_to_move)**：如果智能体停留在起点附近，说明稳定性惩罚过重或 progress_reward 权重不足，需要调整平衡。