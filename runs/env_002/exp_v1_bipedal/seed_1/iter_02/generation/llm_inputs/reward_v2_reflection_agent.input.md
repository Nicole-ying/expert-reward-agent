# ⚠️ 上一版代码验证失败
错误信息：Reward v2 failed validation: runs\env_002\exp_v1_bipedal\seed_1\iter_02\generation\validations\reward_v2.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: 250.895712）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    # 使用 next_obs[2] 作为水平前进速度，鼓励 agent 向前移动
    forward_velocity = next_obs[2]
    # 限制速度范围，避免过大速度导致不稳定
    forward_velocity_clipped = max(-2.0, min(2.0, forward_velocity))
    # 只奖励正向速度（前进），负向速度（后退）不奖励
    forward_reward = 2.0 * max(0.0, forward_velocity_clipped)
    
    # 存活奖励：每步给予小额奖励，鼓励 agent 保持不倒
    # 通过检查 next_obs[0]（主体角度）是否在合理范围内来判断是否存活
    # 角度绝对值小于 1.0 弧度（约 57 度）视为存活
    hull_angle = next_obs[0]
    is_alive = abs(hull_angle) < 1.0
    alive_bonus = 0.5 if is_alive else 0.0
    
    # 稳定性惩罚：轻量约束，抑制过大角度和角速度
    # 使用 next_obs[0]（角度）和 next_obs[1]（角速度）
    angle_penalty = -0.1 * (hull_angle ** 2)
    angular_vel_penalty = -0.05 * (next_obs[1] ** 2)
    stability_penalty = angle_penalty + angular_vel_penalty
    
    # 总奖励
    total_reward = forward_reward + alive_bonus + stability_penalty
    
    # 组件字典
    components = {
        "progress_reward": forward_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=250.895712, len=1357.900000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.497998 | 0.497998 | 0.995996 | 0.867781 |
| progress_reward | 0.573876 | 0.573876 | 0.976599 | 1.000000 |
| stability_penalty | -0.010121 | 0.010121 | 1.000000 | -0.017635 |
| total_reward | 1.061753 | 1.062286 | 1.000000 | 1.850145 |
| generated_reward | 1.061753 | 1.062286 | 1.000000 | 1.850145 |
| original_env_reward | 0.113551 | 0.191083 | 1.000000 | 0.197867 |
| original_env_reward | 0.113551 | 0.191083 | 1.000000 | 0.197867 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=250.895712, min=242.616030, max=257.639807
- episode_length: mean=1357.900000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | alive_bonus + progress_reward + stability_penalty | 250.90 | 250.90 | 0.00 | 1357.90 | alive_bonus=0.498 progress_reward=0.574 stability_penalty=-0.010 | target_solved_new_best |
