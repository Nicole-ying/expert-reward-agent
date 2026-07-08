# 上一轮奖励函数代码（该轮得分: -115.294647）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 目标位置为原点 (0,0)
    # 计算到目标的欧氏距离（使用 next_obs）
    dist_to_target = ((next_obs[0] - 0.0) ** 2 + (next_obs[1] - 0.0) ** 2) ** 0.5

    # 主学习信号：负距离，引导飞行器持续靠近目标（保持不变）
    distance_reward = -dist_to_target

    # 稳定性惩罚：轻量抑制高速、大角度和角速度（保持不变）
    speed_norm = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angvel_abs = abs(next_obs[5])

    lambda_vel = 0.02
    lambda_angle = 0.02
    lambda_av = 0.02

    stability_penalty = -(lambda_vel * speed_norm + lambda_angle * angle_abs + lambda_av * angvel_abs)

    # === 核心改动：软着陆从稀疏二值 → 连续 bounded 乘积 ===
    # 原理：每个因子是 bounded 连续函数，提供稠密梯度
    #   proximity = 1/(1+5*dist)：远处≈0，原点=1，处处有梯度
    #   speed_factor = max(0, 1-speed/0.5)：速度<0.5时有梯度，>=0.5时为0
    #   angle_factor = max(0, 1-angle/0.2)：角度<0.2时有梯度，>=0.2时为0
    #   contact_bonus = (L+R)/2：连续接触信号
    # 乘积后乘 5.0，使理想着陆状态(d≈0,v≈0,a≈0,双触)奖励≈5，与 distance_reward 量级可比
    proximity = 1.0 / (1.0 + 5.0 * dist_to_target)
    speed_factor = max(0.0, 1.0 - speed_norm / 0.5)
    angle_factor = max(0.0, 1.0 - angle_abs / 0.2)
    contact_bonus = (next_obs[6] + next_obs[7]) / 2.0

    soft_landing_continuous = proximity * speed_factor * angle_factor * contact_bonus * 5.0

    # 总奖励
    total_reward = distance_reward + stability_penalty + soft_landing_continuous

    # 记录各组件
    components = {
        'distance_reward': distance_reward,
        'stability_penalty': stability_penalty,
        'soft_landing_continuous': soft_landing_continuous,
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
score=-115.294647, len=68.400000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| distance_reward | -0.969099 | 0.969099 | 1.000000 | 16.837481 |
| soft_landing_continuous | 0.012902 | 0.012902 | 0.005527 | 0.224167 |
| stability_penalty | -0.026946 | 0.026946 | 1.000000 | 0.468169 |
| total_reward | -0.983142 | 1.007392 | 1.000000 | 17.502804 |
| generated_reward | -0.983142 | 1.007392 | 1.000000 | 17.502804 |
| original_env_reward | -1.928712 | 2.731872 | 1.000000 | 47.464575 |

## Distribution
- score: mean=-115.294647, min=-141.582367, max=-95.059093
- episode_length: mean=68.400000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -111.87 | -111.87 | 0.00 | 68.45 | distance_reward=-0.971 soft_landing_proxy=0.028 stability_penalty=-0.026 | new_best |
| 2 | distance_reward + soft_landing_continuous + stability_penalty | -115.29 | -111.87 | -3.42 | 68.40 | distance_reward=-0.969 soft_landing_continuous=0.013 stability_penalty=-0.027 | no_meaningful_improvement |
