# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: runs\env_001\exp_v7\seed_0\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -1368.988936）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # === 势能塑形 (Potential-Based Shaping) ===
    # 原理：F = γ * Φ(s') - Φ(s)，是唯一保证最优策略不变的塑形方式。
    # Φ(s) = -(dist + α*speed + β*|angle|)，同时引导靠近、减速、姿态稳定。
    # 相比 distance_reward = -dist'（只关心绝对位置），势能塑形奖励"改善量"，
    # 提供更稠密的梯度：靠近有奖、减速有奖、摆正有奖。

    # 当前状态
    dist = ((obs[0] - 0.0) ** 2 + (obs[1] - 0.0) ** 2) ** 0.5
    speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    angle = abs(obs[4])

    # 下一状态
    dist_next = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_next = abs(next_obs[4])

    # 势能函数：负的加权状态代价
    alpha_speed = 0.5   # 速度权重，使 speed 分量与 dist 分量量级可比
    beta_angle = 0.5    # 角度权重，使 angle 分量与 dist 分量量级可比
    phi_now = -(dist + alpha_speed * speed + beta_angle * angle)
    phi_next = -(dist_next + alpha_speed * speed_next + beta_angle * angle_next)

    gamma = 0.99  # 接近 1，轻微折扣提供时间偏好
    potential_shaping = gamma * phi_next - phi_now

    # === 独立接触奖励（加法，不被乘积归零） ===
    # 左右腿接触均为 [0,1] 连续值
    contact_bonus = (next_obs[6] + next_obs[7]) * 0.1

    # === 总奖励 ===
    total_reward = potential_shaping + contact_bonus

    components = {
        'potential_shaping': potential_shaping,
        'contact_bonus': contact_bonus,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为原点 (0,0)
    # 计算到目标的欧氏距离（使用 next_obs）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5
    
    # 主学习信号：负距离，引导飞行器持续靠近目标
    distance_reward = -dist_to_target

    # 稳定性惩罚：轻量抑制高速、大角度和角速度，促进安全减速和姿态稳定
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])
    
    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av   = 0.02
    
    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # 软着陆近似奖励：多条件组合，只有当飞行器非常接近目标、速度极低、姿态稳定且双支撑脚均接触时才触发
    threshold_dist   = 0.3
    threshold_speed  = 0.1
    threshold_angle  = 0.05
    contact_left     = next_obs[6]
    contact_right    = next_obs[7]
    
    if (dist_to_target < threshold_dist and 
        speed_norm < threshold_speed and 
        angle_abs < threshold_angle and 
        contact_left == 1.0 and 
        contact_right == 1.0):
        soft_landing_proxy = 10.0
    else:
        soft_landing_proxy = 0.0

    # 总奖励
    total_reward = distance_reward + stability_penalty + soft_landing_proxy

    # 记录各组件
    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-1368.988936, len=177.600000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| contact_bonus | 0.000750 | 0.000750 | 0.004858 | 0.011621 |
| potential_shaping | 0.126059 | 0.126599 | 1.000000 | 1.962533 |
| total_reward | 0.126809 | 0.127322 | 1.000000 | 1.973750 |
| generated_reward | 0.126809 | 0.127322 | 1.000000 | 1.973750 |
| original_env_reward | -5.606683 | 6.068778 | 1.000000 | 94.078345 |

## Distribution
- score: mean=-1368.988936, min=-8273.598986, max=-356.722928
- episode_length: mean=177.600000
- early_terminal (<150 steps + score<-50): 15/20 (75%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -111.87 | -111.87 | 0.00 | 68.45 | distance_reward=-0.971 soft_landing_proxy=0.028 stability_penalty=-0.026 | new_best |
| 2 | distance_reward + soft_landing_continuous + stability_penalty | -115.29 | -111.87 | -3.42 | 68.40 | distance_reward=-0.969 soft_landing_continuous=0.013 stability_penalty=-0.027 | no_meaningful_improvement |
| 3 | contact_bonus + potential_shaping | -1368.99 | -111.87 | -1257.12 | 177.60 | contact_bonus=0.001 potential_shaping=0.126 | no_meaningful_improvement |
