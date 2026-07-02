# 上一轮奖励函数代码（该轮得分: 242.406225）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Calculate distance to target (0,0) for obs and next_obs
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # Primary learning signal: progress towards the landing pad
    progress_reward = 1.0 * (dist_obs - dist_next)

    # Stability penalty with distance-gating.
    # Gate = 1/(1+5*dist): ~1.0 at pad, ~0.17 at dist=1.0, ~0.05 at dist=4.0.
    # This lets the agent move freely when far away, and only tightens
    # speed/angle/angvel constraints as it approaches the landing zone.
    speed_next = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    dist_gate = 1.0 / (1.0 + 5.0 * dist_next)
    stability_penalty = dist_gate * (
        -0.01 * speed_next
        - 0.01 * abs(next_obs[4])
        - 0.005 * abs(next_obs[5])
    )

    # Continuous landing proxy: product of bounded factors provides gradient
    # throughout approach. Each factor max(0, 1 - value/threshold) decays
    # linearly from 1→0 as the dimension worsens.
    near_factor = max(0.0, 1.0 - dist_next / 1.0)
    slow_factor = max(0.0, 1.0 - speed_next / 1.0)
    upright_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.5)
    leg_factor = (next_obs[6] + next_obs[7]) / 2.0

    landing_proxy = near_factor * slow_factor * upright_factor * leg_factor

    total_reward = progress_reward + stability_penalty + landing_proxy

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'dist_gate': dist_gate,
        'landing_proxy': landing_proxy,
        'total_reward': total_reward
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=242.406225, len=481.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| dist_gate | 0.526483 | 0.526483 | 1.000000 | 168.653546 |
| landing_proxy | 0.447489 | 0.447489 | 0.561646 | 143.348474 |
| progress_reward | 0.003122 | 0.003423 | 0.999584 | 1.000000 |
| stability_penalty | -0.001045 | 0.001045 | 1.000000 | -0.334648 |
| total_reward | 0.449566 | 0.450037 | 1.000000 | 144.013826 |
| generated_reward | 0.449566 | 0.450037 | 1.000000 | 144.013826 |
| original_env_reward | -0.038528 | 1.560899 | 1.000000 | -12.342087 |

## Distribution
- score: mean=242.406225, min=109.652525, max=282.466855
- episode_length: mean=481.800000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_proxy + progress_reward + stability_penalty | -111.71 | -111.71 | 0.00 | 69.10 | landing_proxy=0.005 progress_reward=0.016 stability_penalty=-0.011 | new_best |
| 2 | landing_proxy + progress_reward + stability_penalty | -89.18 | -89.18 | 0.00 | 70.10 | landing_proxy=0.005 progress_reward=0.016 stability_penalty=-0.011 | new_best |
| 3 | dist_gate + landing_proxy + progress_reward + stability_penalty | 242.41 | 242.41 | 0.00 | 481.80 | dist_gate=0.526 landing_proxy=0.447 progress_reward=0.003 stability_penalty=-0.001 | target_solved_new_best |
