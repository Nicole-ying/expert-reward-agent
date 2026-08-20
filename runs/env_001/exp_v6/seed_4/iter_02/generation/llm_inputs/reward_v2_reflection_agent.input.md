# 上一轮奖励函数代码（该轮得分: -111.441730）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：进度差分奖励（接近目标垫中心）
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # 正值表示接近目标

    # 2. 稳定性约束：惩罚高速、大倾角和高角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angular_speed = abs(next_obs[5])

    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.05
    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angular_speed)

    # 3. 任务完成代理：软着陆近似奖励
    near_target = (dist_next < 0.2)
    low_speed = (speed < 0.1)
    stable_angle = (angle_abs < 0.1)
    both_contact = (next_obs[6] == 1.0 and next_obs[7] == 1.0)
    soft_landing = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_delta + stability_penalty + soft_landing

    components = {
        "progress": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-111.441730, len=71.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.016054 | 0.016993 | 0.999993 | 0.016054 |
| soft_landing_proxy | 0.000777 | 0.000777 | 0.003883 | 0.000777 |
| stability_penalty | -0.057148 | 0.057148 | 1.000000 | -0.057148 |
| total_reward | -0.040317 | 0.041800 | 1.000000 | -0.040317 |
| generated_reward | -0.040317 | 0.041800 | 1.000000 | -0.040317 |
| original_env_reward | -1.548622 | 2.378699 | 1.000000 | -1.548622 |

## Distribution
- score: mean=-111.441730, min=-122.161769, max=-95.059093
- episode_length: mean=71.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -111.44 | -111.44 | 0.00 | 71.30 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.057 | new_best |
