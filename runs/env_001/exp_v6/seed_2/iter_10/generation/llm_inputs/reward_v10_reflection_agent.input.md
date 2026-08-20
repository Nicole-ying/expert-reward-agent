# 上一轮奖励函数代码（该轮得分: 251.358113）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Stage-weighted: early focus on proximity, late focus on precision landing.
    No leg-contact checks — avoids terminal_success_reward proxy.
    """
    # --- hyperparameters ---
    k_proximity = 5.0          # decay rate for bounded proximity
    w_proximity_early = 3.0    # proximity weight in early training
    w_quality_late = 4.0       # landing quality weight in late training
    w_stability = 0.02         # stability penalty weight (mild, background)

    # thresholds for continuous landing quality factors
    D_near = 0.5               # distance: full quality at 0, zero beyond 0.5
    V_slow = 0.5               # speed: full quality at 0, zero beyond 0.5 m/s
    A_upright = 0.3            # angle: full quality at 0, zero beyond 0.3 rad

    # --- extract state ---
    dx, dy = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    dist = (dx * dx + dy * dy) ** 0.5
    speed = (vx * vx + vy * vy) ** 0.5

    # --- 1. bounded proximity reward (dense, always active) ---
    proximity = 1.0 / (1.0 + k_proximity * dist)

    # --- 2. continuous landing quality (no leg contacts, purely geometric) ---
    near_factor = max(0.0, 1.0 - dist / D_near)
    slow_factor = max(0.0, 1.0 - speed / V_slow)
    upright_factor = max(0.0, 1.0 - abs(angle) / A_upright)
    landing_quality = near_factor * slow_factor * upright_factor

    # --- 3. mild stability penalty (angular velocity only, to avoid spinning) ---
    stability_penalty = -w_stability * abs(ang_vel)

    # --- stage weighting ---
    # early: proximity dominates → agent learns to approach target
    # late:  landing quality dominates → agent learns precision touchdown
    early_w = max(0.0, 1.0 - training_progress)
    late_w = training_progress

    progress_signal = early_w * w_proximity_early * proximity
    quality_signal = late_w * w_quality_late * landing_quality

    total_reward = progress_signal + quality_signal + stability_penalty

    components = {
        "proximity": proximity,
        "landing_quality": landing_quality,
        "stability_penalty": stability_penalty,
        "progress_signal": progress_signal,
        "quality_signal": quality_signal,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=251.358113, len=375.700000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| landing_quality | 0.318642 | 0.318642 | 0.583516 | 0.318642 |
| progress_signal | 1.186551 | 1.186551 | 1.000000 | 1.186551 |
| proximity | 0.461166 | 0.461166 | 1.000000 | 0.461166 |
| quality_signal | 0.204100 | 0.204100 | 0.583516 | 0.204100 |
| stability_penalty | -0.002367 | 0.002367 | 0.999999 | -0.002367 |
| total_reward | 1.388284 | 1.388284 | 1.000000 | 1.388284 |
| generated_reward | 1.388284 | 1.388284 | 1.000000 | 1.388284 |
| original_env_reward | -0.160388 | 3.418529 | 1.000000 | -0.160388 |

## Distribution
- score: mean=251.358113, min=217.665196, max=276.969335
- episode_length: mean=375.700000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
| 2 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -98.04 | -98.04 | 0.00 | 72.30 | distance_penalty=-0.005 progress_delta_reward=0.016 soft_landing_bonus=0.009 stability_penalty=-0.009 | new_best |
| 3 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -12.26 | -12.26 | 0.00 | 1000.00 | approach_bonus=2.711 distance_penalty=-0.003 progress_delta_reward=0.002 stability_penalty=-0.005 | new_best |
| 4 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -60.67 | -12.26 | -48.41 | 85.30 | approach_bonus=0.005 distance_penalty=-0.004 progress_delta_reward=0.013 stability_penalty=-0.007 | no_meaningful_improvement |
| 5 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -110.31 | -12.26 | -98.05 | 72.00 | approach_bonus=0.009 distance_penalty=-0.005 progress_delta_reward=0.016 stability_penalty=-0.008 | no_meaningful_improvement |
| 6 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -55.92 | -12.26 | -43.65 | 955.10 | approach_bonus=3.129 distance_penalty=-0.002 progress_delta_reward=0.002 stability_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 7 | potential_shaping + soft_landing_proxy + stability_penalty | -108.75 | -108.75 | 0.00 | 72.00 | potential_shaping=0.021 soft_landing_proxy=0.010 stability_penalty=-0.031 | new_best |
| 8 | potential_shaping + soft_landing_proxy + stability_penalty | -0.83 | -0.83 | 0.00 | 1000.00 | potential_shaping=0.005 soft_landing_proxy=0.917 stability_penalty=-0.015 | new_best |
| 9 | landing_quality + progress_signal + proximity + quality_signal + stability_penalty | 251.36 | 251.36 | 0.00 | 375.70 | landing_quality=0.319 progress_signal=1.187 proximity=0.461 quality_signal=0.204 stability_penalty=-0.002 | target_solved_new_best |
