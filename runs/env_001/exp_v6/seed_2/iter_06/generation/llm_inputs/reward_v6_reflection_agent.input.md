# 上一轮奖励函数代码（该轮得分: -110.309295）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (unchanged) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (unchanged) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- approach quality bonus: potential-based shaping ----
    # KEY CHANGE: switched from state-reward (product form → hover exploit)
    # to potential-based shaping (Φ(next) - Φ(obs)).
    # Φ = proximity * speed_ok * angle_ok — a bounded [0,1] potential that
    # captures "how good is the approach state".
    # Bonus = Φ(next) - Φ(obs) — only rewards IMPROVEMENT, zero when static.
    # This naturally prevents the hover exploit: staying still gives ΔΦ = 0.
    APPROACH_DIST = 2.0
    SPEED_THRESH = 2.0
    ANGLE_THRESH = 0.5

    # Potential at current state (obs)
    prox_now = max(0.0, 1.0 - d_obs / APPROACH_DIST)
    spd_now = max(0.0, 1.0 - (abs(obs[2]) + abs(obs[3])) / SPEED_THRESH)
    ang_now = max(0.0, 1.0 - abs(obs[4]) / ANGLE_THRESH)
    potential_now = prox_now * spd_now * ang_now

    # Potential at next state (next_obs)
    prox_next = max(0.0, 1.0 - d_next / APPROACH_DIST)
    spd_next = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    ang_next = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)
    potential_next = prox_next * spd_next * ang_next

    # γ = 1.0 (standard for potential-based shaping, per Ng 1999)
    approach_bonus = (potential_next - potential_now) * 2.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + approach_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "approach_bonus": approach_bonus,
        "total_reward": total,
    }
    return float(total), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal, unchanged) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (unchanged) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (unchanged) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- continuous approach bonus (REMOVED both_legs_contact binary gate) ----
    # proximity provides natural distance-gating: zero when far, gradient when close.
    # speed and angle factors give gradient toward "slow + upright" approach.
    APPROACH_DIST = 2.0      # bonus activates within this radius
    SPEED_THRESH = 2.0       # combined |vx|+|vy| threshold
    ANGLE_THRESH = 0.5       # tilt threshold

    proximity = max(0.0, 1.0 - d_next / APPROACH_DIST)
    speed_ok = max(0.0, 1.0 - (abs(vx) + abs(vy)) / SPEED_THRESH)
    angle_ok = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # Product of 3 bounded [0,1] factors, scaled so a good approach gives
    # meaningful reward.  No binary gate — gradient flows at all distances < APPROACH_DIST.
    approach_bonus = proximity * speed_ok * angle_ok * 5.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + approach_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "approach_bonus": approach_bonus,
        "total_reward": total,
    }
    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-110.309295, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| approach_bonus | 0.008636 | 0.020279 | 0.894846 | 0.548385 |
| distance_penalty | -0.004836 | 0.004836 | 1.000000 | -0.307105 |
| progress_delta_reward | 0.015748 | 0.016686 | 0.999993 | 1.000000 |
| stability_penalty | -0.008421 | 0.008421 | 1.000000 | -0.534753 |
| total_reward | 0.011126 | 0.020762 | 1.000000 | 0.706527 |
| generated_reward | 0.011126 | 0.020762 | 1.000000 | 0.706527 |
| original_env_reward | -1.419141 | 2.457787 | 1.000000 | -90.116457 |

## Distribution
- score: mean=-110.309295, min=-122.412495, max=-98.654419
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
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
