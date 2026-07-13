# 上一轮奖励函数代码（该轮得分: 250.403608）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    # 使用水平速度(obs[2])作为前进驱动力
    forward_velocity = next_obs[2]
    fwd_scale = 3.0  # 从2.0提高到3.0，增强前进驱动力
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：连续化，鼓励稳定站立 ==========
    hull_angle = abs(next_obs[0])
    hull_angular_vel = abs(next_obs[1])
    
    # 连续存活奖励：角度和角速度都接近0时拿满，越接近边界越小
    # 使用二次衰减，当角度=0且角速度=0时 reward=0.2
    # 当角度接近0.5或角速度接近2.0时 reward→0
    angle_factor = 1.0 - (hull_angle / 0.5) ** 2
    vel_factor = 1.0 - (hull_angular_vel / 2.0) ** 2
    # 限制在[0,1]范围，避免负值
    angle_factor = max(0.0, angle_factor)
    vel_factor = max(0.0, vel_factor)
    
    alive_bonus = 0.2 * angle_factor * vel_factor  # 连续值，最大0.2
    
    # ========== 稳定性约束：适度惩罚 ==========
    # 惩罚过大的主体角度和角速度
    angle_penalty_scale = 1.0    # 从0.5提高到1.0
    angular_vel_penalty_scale = 0.5  # 从0.3提高到0.5
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    # 使用水平速度(obs[2])作为前进驱动力
    # 鼓励agent向前移动，这是任务的核心目标
    forward_velocity = next_obs[2]  # 使用next_obs避免延迟
    fwd_scale = 2.0
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：鼓励持续行走 ==========
    # 通过主体角度(obs[0])和角速度(obs[1])判断是否还站立
    # 当主体角度过大或角速度过高时，说明即将摔倒，减少存活奖励
    hull_angle = abs(next_obs[0])  # 主体偏离竖直方向的角度
    hull_angular_vel = abs(next_obs[1])  # 主体角速度绝对值
    
    # 存活条件：角度小于0.5弧度(~28度)且角速度小于2.0
    alive_condition = (hull_angle < 0.5) and (hull_angular_vel < 2.0)
    alive_bonus = 0.5 if alive_condition else 0.0
    
    # ========== 稳定性约束：轻量惩罚 ==========
    # 惩罚过大的主体角度和角速度，防止摔倒
    # 使用连续函数，避免二值惩罚导致的梯度消失
    angle_penalty_scale = 0.5
    angular_vel_penalty_scale = 0.3
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=250.403608, len=1077.400000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.174791 | 0.174791 | 0.989832 | 0.189703 |
| progress_reward | 0.921391 | 0.926730 | 1.000000 | 1.000000 |
| stability_penalty | -0.036376 | 0.036376 | 1.000000 | -0.039480 |
| total_reward | 1.059806 | 1.069011 | 1.000000 | 1.150224 |
| generated_reward | 1.059806 | 1.069011 | 1.000000 | 1.150224 |
| original_env_reward | 0.131734 | 0.189971 | 1.000000 | 0.142973 |
| original_env_reward | 0.131734 | 0.189971 | 1.000000 | 0.142973 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=250.403608, min=-49.408814, max=281.502100
- episode_length: mean=1077.400000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | alive_bonus + progress_reward + stability_penalty | 244.95 | 244.95 | 0.00 | 1385.75 | alive_bonus=0.497 progress_reward=0.572 stability_penalty=-0.015 | new_best |
| 2 | alive_bonus + progress_reward + stability_penalty | 250.40 | 244.95 | 5.46 | 1077.40 | alive_bonus=0.175 progress_reward=0.921 stability_penalty=-0.036 | target_solved_no_improvement |
