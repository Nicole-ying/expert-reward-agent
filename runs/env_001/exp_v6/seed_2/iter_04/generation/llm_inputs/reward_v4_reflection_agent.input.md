# 上一轮奖励函数代码（该轮得分: -12.262791）
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
score=-12.262791, len=1000.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| approach_bonus | 2.711457 | 2.711457 | 0.983185 | 1584.587440 |
| distance_penalty | -0.002674 | 0.002674 | 1.000000 | -1.562526 |
| progress_delta_reward | 0.001711 | 0.002434 | 0.999895 | 1.000000 |
| stability_penalty | -0.005310 | 0.005310 | 1.000000 | -3.103214 |
| total_reward | 2.705185 | 2.706028 | 1.000000 | 1580.921700 |
| generated_reward | 2.705185 | 2.706028 | 1.000000 | 1580.921700 |
| original_env_reward | -0.086158 | 2.679535 | 1.000000 | -50.350950 |

## Distribution
- score: mean=-12.262791, min=-40.057329, max=23.871013
- episode_length: mean=1000.000000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
| 2 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -98.04 | -98.04 | 0.00 | 72.30 | distance_penalty=-0.005 progress_delta_reward=0.016 soft_landing_bonus=0.009 stability_penalty=-0.009 | new_best |
| 3 | approach_bonus + distance_penalty + progress_delta_reward + stability_penalty | -12.26 | -12.26 | 0.00 | 1000.00 | approach_bonus=2.711 distance_penalty=-0.003 progress_delta_reward=0.002 stability_penalty=-0.005 | new_best |
