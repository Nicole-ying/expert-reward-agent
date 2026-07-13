# 上一轮奖励函数代码（该轮得分: 263.365361）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    forward_velocity = next_obs[2]
    fwd_scale = 2.5  # 从3.0降到2.5，平衡速度与稳定性
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：连续化，鼓励稳定站立 ==========
    hull_angle = abs(next_obs[0])
    hull_angular_vel = abs(next_obs[1])
    
    # 连续存活奖励：角度和角速度都接近0时拿满
    angle_factor = 1.0 - (hull_angle / 0.5) ** 2
    vel_factor = 1.0 - (hull_angular_vel / 2.0) ** 2
    angle_factor = max(0.0, angle_factor)
    vel_factor = max(0.0, vel_factor)
    
    alive_bonus = 0.5 * angle_factor * vel_factor  # 从0.2提高到0.5，增强存活引导
    
    # ========== 稳定性约束：适度惩罚 ==========
    angle_penalty_scale = 1.0
    angular_vel_penalty_scale = 0.5
    
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
score=263.365361, len=1318.800000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.465762 | 0.465762 | 0.993467 | 0.641831 |
| progress_reward | 0.725677 | 0.729534 | 1.000000 | 1.000000 |
| stability_penalty | -0.021328 | 0.021328 | 1.000000 | -0.029391 |
| total_reward | 1.170111 | 1.176911 | 1.000000 | 1.612440 |
| generated_reward | 1.170111 | 1.176911 | 1.000000 | 1.612440 |
| original_env_reward | 0.112716 | 0.177540 | 1.000000 | 0.155325 |
| original_env_reward | 0.112716 | 0.177540 | 1.000000 | 0.155325 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=263.365361, min=260.325505, max=266.465812
- episode_length: mean=1318.800000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | alive_bonus + progress_reward + stability_penalty | 244.95 | 244.95 | 0.00 | 1385.75 | alive_bonus=0.497 progress_reward=0.572 stability_penalty=-0.015 | new_best |
| 2 | alive_bonus + progress_reward + stability_penalty | 250.40 | 244.95 | 5.46 | 1077.40 | alive_bonus=0.175 progress_reward=0.921 stability_penalty=-0.036 | target_solved_no_improvement |
| 3 | alive_bonus + progress_reward + stability_penalty | 263.37 | 263.37 | 0.00 | 1318.80 | alive_bonus=0.466 progress_reward=0.726 stability_penalty=-0.021 | target_solved_new_best |
