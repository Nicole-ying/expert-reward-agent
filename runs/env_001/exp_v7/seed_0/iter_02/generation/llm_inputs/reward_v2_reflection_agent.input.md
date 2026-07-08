# 上一轮奖励函数代码（该轮得分: -111.870902）
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
score=-111.870902, len=68.450000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| distance_reward | -0.970523 | 0.970523 | 1.000000 | 17.112529 |
| soft_landing_proxy | 0.028500 | 0.028500 | 0.002850 | 0.502514 |
| stability_penalty | -0.025649 | 0.025649 | 1.000000 | 0.452248 |
| total_reward | -0.967673 | 1.024070 | 1.000000 | 18.056669 |
| generated_reward | -0.967673 | 1.024070 | 1.000000 | 18.056669 |
| original_env_reward | -1.750103 | 2.598609 | 1.000000 | 45.819371 |

## Distribution
- score: mean=-111.870902, min=-125.263486, max=-95.638526
- episode_length: mean=68.450000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -111.87 | -111.87 | 0.00 | 68.45 | distance_reward=-0.971 soft_landing_proxy=0.028 stability_penalty=-0.026 | new_best |
