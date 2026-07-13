# 上一轮奖励函数代码（该轮得分: 251.914036）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：前进速度奖励
    forward_velocity = next_obs[2]  # horizontal_velocity
    progress_reward = 2.0 * forward_velocity
    
    # 稳定性约束：惩罚身体倾斜和角速度
    hull_angle = next_obs[0]  # hull_angle
    hull_angular_velocity = next_obs[1]  # hull_angular_velocity
    stability_penalty = -0.5 * (hull_angle ** 2) - 0.1 * (hull_angular_velocity ** 2)
    
    # 步态质量奖励：鼓励交替接触地面
    leg1_contact = next_obs[12]  # leg1_contact
    leg2_contact = next_obs[13]  # leg2_contact
    # 当两条腿交替接触时给予奖励（异或逻辑）
    gait_quality = abs(leg1_contact - leg2_contact)  # 1.0 when exactly one leg contacts
    gait_bonus = 0.3 * gait_quality
    
    # 总奖励
    total_reward = progress_reward + stability_penalty + gait_bonus
    
    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "gait_bonus": gait_bonus
    }
    
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=251.914036, len=1195.850000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| gait_bonus | 0.346973 | 0.346973 | 0.996762 | 0.576391 |
| progress_reward | 0.601975 | 0.605476 | 1.000000 | 1.000000 |
| stability_penalty | -0.040546 | 0.040546 | 1.000000 | -0.067355 |
| total_reward | 0.908402 | 0.910606 | 1.000000 | 1.509036 |
| generated_reward | 0.908402 | 0.910606 | 1.000000 | 1.509036 |
| original_env_reward | 0.121821 | 0.197875 | 1.000000 | 0.202368 |
| original_env_reward | 0.121821 | 0.197875 | 1.000000 | 0.202368 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=251.914036, min=76.665034, max=264.748147
- episode_length: mean=1195.850000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | alive_bonus + progress_reward + stability_penalty | 244.95 | 244.95 | 0.00 | 1385.75 | alive_bonus=0.497 progress_reward=0.572 stability_penalty=-0.015 | new_best |
| 2 | alive_bonus + progress_reward + stability_penalty | 250.40 | 244.95 | 5.46 | 1077.40 | alive_bonus=0.175 progress_reward=0.921 stability_penalty=-0.036 | target_solved_no_improvement |
| 3 | alive_bonus + progress_reward + stability_penalty | 263.37 | 263.37 | 0.00 | 1318.80 | alive_bonus=0.466 progress_reward=0.726 stability_penalty=-0.021 | target_solved_new_best |
| 4 | alive_bonus + progress_reward + stability_penalty | 269.69 | 263.37 | 6.32 | 1279.35 | alive_bonus=0.455 progress_reward=1.180 stability_penalty=-0.025 | target_solved_no_improvement |
| 6 | gait_bonus + progress_reward + stability_penalty | 251.91 | 251.91 | 0.00 | 1195.85 | gait_bonus=0.347 progress_reward=0.602 stability_penalty=-0.041 | target_solved_new_best |
