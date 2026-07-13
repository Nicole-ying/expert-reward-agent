# 上一轮奖励函数代码（该轮得分: 244.945917）
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
score=244.945917, len=1385.750000, errors=0

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | ratio_to_progress_reward |
|-----------|------|----------|-------------|--------------------------|
| alive_bonus | 0.497027 | 0.497027 | 0.994054 | 0.869421 |
| progress_reward | 0.571676 | 0.575681 | 1.000000 | 1.000000 |
| stability_penalty | -0.015351 | 0.015351 | 1.000000 | -0.026853 |
| total_reward | 1.053352 | 1.056374 | 1.000000 | 1.842568 |
| generated_reward | 1.053352 | 1.056374 | 1.000000 | 1.842568 |
| original_env_reward | 0.117302 | 0.175417 | 1.000000 | 0.205189 |
| original_env_reward | 0.117302 | 0.175417 | 1.000000 | 0.205189 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Distribution
- score: mean=244.945917, min=242.129347, max=247.501914
- episode_length: mean=1385.750000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | alive_bonus + progress_reward + stability_penalty | 244.95 | 244.95 | 0.00 | 1385.75 | alive_bonus=0.497 progress_reward=0.572 stability_penalty=-0.015 | new_best |
