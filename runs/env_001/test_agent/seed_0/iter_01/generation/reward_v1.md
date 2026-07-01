# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：progress_delta_reward ==========
    # 目标位置假设为 (0, 0)，因为 obs[0] 和 obs[1] 是相对于目标着陆平台的坐标
    # 计算当前距离平方和下一时刻距离平方
    current_dist_sq = obs[0] ** 2 + obs[1] ** 2
    next_dist_sq = next_obs[0] ** 2 + next_obs[1] ** 2
    
    # progress_delta: 正数表示更接近目标
    progress_delta = current_dist_sq - next_dist_sq
    
    # 缩放因子，使奖励值在合理范围
    progress_scale = 2.0
    progress_delta_reward = progress_scale * progress_delta
    
    # ========== 稳定/安全约束：stability_penalty ==========
    # 惩罚高速、大姿态角和大角速度
    # 使用 next_obs 因为动作效果体现在下一状态
    vel_x = next_obs[2]
    vel_y = next_obs[3]
    body_angle = next_obs[4]
    angular_vel = next_obs[5]
    
    # 速度惩罚：鼓励减速接近目标
    speed = (vel_x ** 2 + vel_y ** 2) ** 0.5
    speed_penalty_weight = 0.3
    speed_penalty = -speed_penalty_weight * speed
    
    # 姿态角惩罚：鼓励保持水平姿态（角度接近0）
    angle_penalty_weight = 0.2
    angle_penalty = -angle_penalty_weight * abs(body_angle)
    
    # 角速度惩罚：鼓励稳定姿态
    angular_vel_penalty_weight = 0.1
    angular_vel_penalty = -angular_vel_penalty_weight * abs(angular_vel)
    
    stability_penalty = speed_penalty + angle_penalty + angular_vel_penalty
    
    # ========== 任务完成 proxy：soft_landing_proxy ==========
    # 当飞行器接近目标、速度低、姿态稳定且双支撑接触时给予小奖励
    # 条件：距离近、速度低、姿态角小、双接触
    near_target_threshold = 0.5
    low_speed_threshold = 0.3
    stable_angle_threshold = 0.2
    
    is_near_target = current_dist_sq ** 0.5 < near_target_threshold
    is_low_speed = speed < low_speed_threshold
    is_stable_angle = abs(body_angle) < stable_angle_threshold
    is_both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)
    
    soft_landing_bonus = 0.0
    if is_near_target and is_low_speed and is_stable_angle and is_both_contact:
        soft_landing_bonus = 1.0
    
    soft_landing_proxy = soft_landing_bonus
    
    # ========== 组合总奖励 ==========
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy
    
    # ========== 构建 components dict ==========
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_delta_reward**（主学习信号）
   - 角色：引导飞行器向目标移动，每一步更接近目标获得正奖励
   - 使用 obs[0], obs[1] 和 next_obs[0], next_obs[1] 计算距离平方差
   - 这是核心学习信号，提供密集的进度引导

2. **stability_penalty**（稳定/安全约束）
   - 角色：惩罚高速、大姿态角和大角速度，鼓励稳定接近目标
   - 包含三个子项：速度惩罚、姿态角惩罚、角速度惩罚
   - 权重较小（0.3, 0.2, 0.1），避免过度约束导致不敢移动

3. **soft_landing_proxy**（任务完成 proxy）
   - 角色：当飞行器同时满足接近目标、低速、稳定姿态和双接触时给予小奖励
   - 权重为1.0，仅作为辅助信号，不是主要学习目标
   - 条件严格，避免 reward hacking

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- `explicit_success_flag_available=false`：没有明确的成功标志
- `explicit_failure_flag_available=false`：没有明确的失败标志
- 使用这些信号需要依赖 info 字段或终止条件，但当前环境 step 返回空字典，无法可靠判断成功/失败
- 强行使用可能导致奖励函数不稳定或误导学习

## 留到后续迭代的组件

- **energy_penalty**：当前动作空间是离散的，且任务初期需要鼓励探索，过早加入燃料惩罚可能导致 agent 不敢使用引擎
- **time_penalty**：可能鼓励冒险行为，等 agent 学会稳定接近后再加入
- **gated_reward**：复杂门控逻辑，v1 保持简单
- **terminal_success_reward / terminal_failure_penalty**：等待 wrapper 明确暴露 success/failure 信号

## 训练后应观察的 failure mode

1. **goal_near_oscillation**：飞行器在目标附近震荡，无法稳定着陆
   - 观察：progress_delta_reward 在目标附近正负交替
   - 对策：后续可加入 time_penalty 或增大 stability_penalty 权重

2. **high_reward_without_success**：飞行器获得高奖励但未成功着陆
   - 观察：soft_landing_proxy 频繁触发但实际未稳定着陆
   - 对策：收紧 soft_landing_proxy 条件或移除

3. **fast_crash_near_goal**：飞行器快速接近目标但坠毁
   - 观察：progress_delta_reward 很高但 episode 提前结束
   - 对策：增大速度惩罚或加入姿态约束