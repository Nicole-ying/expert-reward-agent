# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: runs\env_001\exp_v6\seed_4\iter_03\generation\validations\reward_v3.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -111.116530）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：进度差分奖励（接近目标垫中心）
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # 正值表示接近目标

    # 2. 稳定性约束：惩罚高速、大倾角和高角速度
    #    【本轮修改】系数从 0.05 降到 0.005（10x），解决 penalty ratio -3.56 压倒 progress 的问题
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angular_speed = abs(next_obs[5])

    w_speed = 0.005
    w_angle = 0.005
    w_angvel = 0.005
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

# 历史最佳奖励函数代码（历史最高得分）
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
score=-111.116530, len=71.400000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress | 0.016060 | 0.016981 | 0.999993 | 0.016060 |
| soft_landing_proxy | 0.000751 | 0.000751 | 0.003754 | 0.000751 |
| stability_penalty | -0.005795 | 0.005795 | 1.000000 | -0.005795 |
| total_reward | 0.011015 | 0.013335 | 1.000000 | 0.011015 |
| generated_reward | 0.011015 | 0.013335 | 1.000000 | 0.011015 |
| original_env_reward | -1.524781 | 2.427536 | 1.000000 | -1.524781 |

## Distribution
- score: mean=-111.116530, min=-122.935118, max=-95.059093
- episode_length: mean=71.400000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress + soft_landing_proxy + stability_penalty | -111.44 | -111.44 | 0.00 | 71.30 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.057 | new_best |
| 2 | progress + soft_landing_proxy + stability_penalty | -111.12 | -111.44 | 0.33 | 71.40 | progress=0.016 soft_landing_proxy=0.001 stability_penalty=-0.006 | no_meaningful_improvement |
