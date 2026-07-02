# 上一轮奖励函数代码（该轮得分: -89.182935）
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

    # Continuous landing proxy: product of bounded factors provides gradient throughout approach.
    # Each factor max(0, 1 - value/threshold) decays linearly from 1→0 as the dimension worsens.
    # This replaces the binary if-condition that had 0.46% trigger rate with a signal that
    # activates as soon as the agent enters the approach zone.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)       # active within 1.0 units of pad
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)      # rewards speed < 1.0
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)  # rewards |angle| < 0.5
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0       # both legs down = 1.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

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
score=-89.182935, len=70.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_proxy | 0.004930 | 0.004930 | 0.011068 | 0.308166 |
| progress_reward | 0.015997 | 0.016930 | 0.999996 | 1.000000 |
| stability_penalty | -0.011363 | 0.011363 | 1.000000 | -0.710304 |
| total_reward | 0.009564 | 0.013537 | 1.000000 | 0.597862 |
| generated_reward | 0.009564 | 0.013537 | 1.000000 | 0.597862 |
| original_env_reward | -1.510221 | 2.485527 | 1.000000 | -94.404477 |

## Distribution
- score: mean=-89.182935, min=-117.722997, max=-52.225025
- episode_length: mean=70.100000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_proxy + progress_reward + stability_penalty | -111.71 | -111.71 | 0.00 | 69.10 | landing_proxy=0.005 progress_reward=0.016 stability_penalty=-0.011 | new_best |
| 2 | landing_proxy + progress_reward + stability_penalty | -89.18 | -89.18 | 0.00 | 70.10 | landing_proxy=0.005 progress_reward=0.016 stability_penalty=-0.011 | new_best |
