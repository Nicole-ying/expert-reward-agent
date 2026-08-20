# 上一轮奖励函数代码（该轮得分: -108.944442）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：progress delta（到目标垫中心的距离减少量）
    x_prev, y_prev = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    d_prev = (x_prev**2 + y_prev**2) ** 0.5
    d_next = (x_next**2 + y_next**2) ** 0.5
    progress_delta = d_prev - d_next

    # 稳定约束：惩罚速度、角度与角速度（作用于 next_obs）
    x_vel, y_vel = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    w_vel = 0.1
    w_angle = 0.5
    w_angvel = 0.1
    stability_penalty = (
        w_vel * (abs(x_vel) + abs(y_vel)) +
        w_angle * abs(body_angle) +
        w_angvel * abs(ang_vel)
    )

    # 任务完成软代理：接近中心 + 低速 + 水平 + 双脚着地
    near = (d_next < 0.1)
    slow = (abs(x_vel) < 0.1 and abs(y_vel) < 0.1)
    level = (abs(body_angle) < 0.1)
    both_feet = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    soft_landing_bonus_raw = 1.0 if (near and slow and level and both_feet) else 0.0
    soft_landing_bonus = 0.5 * soft_landing_bonus_raw

    total_reward = progress_delta - stability_penalty + soft_landing_bonus

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": soft_landing_bonus,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-108.944442, len=73.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.016063 | 0.017000 | 0.999990 | 1.000000 |
| soft_landing_bonus | 0.001067 | 0.001067 | 0.002134 | 0.066441 |
| stability_penalty | 0.147474 | 0.147474 | 1.000000 | 9.181011 |
| total_reward | -0.130344 | 0.132426 | 1.000000 | -8.114570 |
| generated_reward | -0.130344 | 0.132426 | 1.000000 | -8.114570 |
| original_env_reward | -1.519302 | 2.389056 | 1.000000 | -94.584410 |

## Distribution
- score: mean=-108.944442, min=-121.621397, max=-95.428352
- episode_length: mean=73.600000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_bonus + stability_penalty | -108.94 | -108.94 | 0.00 | 73.60 | progress_delta=0.016 soft_landing_bonus=0.001 stability_penalty=0.147 | new_best |
