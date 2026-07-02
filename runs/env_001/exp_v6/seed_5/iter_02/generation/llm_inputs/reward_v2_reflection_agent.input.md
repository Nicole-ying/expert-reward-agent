# 上一轮奖励函数代码（该轮得分: -111.710629）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty: discourage high speeds, large angle, and angular velocity in the new state
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    stability_penalty = -0.01 * speed_next - 0.01 * abs(next_obs[4]) - 0.005 * abs(next_obs[5])

    # Soft landing proxy: small bonus when close, slow, upright and both supports in contact
    landing_proxy = 0.0
    if (dist_next < 0.3 and speed_next < 0.3 and abs(next_obs[4]) < 0.2 and
        next_obs[6] == 1.0 and next_obs[7] == 1.0):
        landing_proxy = 1.0

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-111.710629, len=69.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_proxy | 0.004575 | 0.004575 | 0.004575 | 0.283420 |
| progress_reward | 0.016142 | 0.017068 | 0.999998 | 1.000000 |
| stability_penalty | -0.011355 | 0.011355 | 1.000000 | -0.703448 |
| total_reward | 0.009362 | 0.013461 | 1.000000 | 0.579972 |
| generated_reward | 0.009362 | 0.013461 | 1.000000 | 0.579972 |
| original_env_reward | -1.582544 | 2.418336 | 1.000000 | -98.040334 |

## Distribution
- score: mean=-111.710629, min=-125.885489, max=-97.868960
- episode_length: mean=69.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_proxy + progress_reward + stability_penalty | -111.71 | -111.71 | 0.00 | 69.10 | landing_proxy=0.005 progress_reward=0.016 stability_penalty=-0.011 | new_best |
