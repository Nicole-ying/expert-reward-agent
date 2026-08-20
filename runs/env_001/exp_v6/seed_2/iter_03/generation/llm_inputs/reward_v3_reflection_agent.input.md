# 上一轮奖励函数代码（该轮得分: -98.037542）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta (main learning signal) ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (anchor, drastically reduced) ----
    distance_penalty = -0.005 * d_next

    # ---- stability penalty (drastically reduced: 10x smaller) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.005 * abs(vx)
        - 0.005 * abs(vy)
        - 0.02 * abs(angle)
        - 0.02 * abs(ang_vel)
    )

    # ---- continuous soft landing proxy (replaces binary bonus) ----
    # Use bounded continuous factors so agent gets gradient as it approaches
    # a good landing, not just at the exact moment of contact.
    LANDING_DIST_THRESH = 1.0
    VEL_THRESH = 0.5
    ANGLE_THRESH = 0.3

    proximity_factor = max(0.0, 1.0 - d_next / LANDING_DIST_THRESH)
    vel_x_factor = max(0.0, 1.0 - abs(vx) / VEL_THRESH)
    vel_y_factor = max(0.0, 1.0 - abs(vy) / VEL_THRESH)
    angle_factor = max(0.0, 1.0 - abs(angle) / ANGLE_THRESH)

    # Only reward when both legs are on the ground (contact), but use
    # continuous factors so the approach path also gets shaped.
    both_legs_contact = 1.0 if (next_obs[6] == 1.0 and next_obs[7] == 1.0) else 0.0

    landing_bonus = (
        proximity_factor
        * vel_x_factor
        * vel_y_factor
        * angle_factor
        * both_legs_contact
        * 2.0  # scale factor so a perfect landing gives meaningful reward
    )

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + landing_bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": landing_bonus,
        "total_reward": total,
    }
    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-98.037542, len=72.300000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_penalty | -0.004847 | 0.004847 | 1.000000 | -0.305614 |
| progress_delta_reward | 0.015859 | 0.016807 | 0.999991 | 1.000000 |
| soft_landing_bonus | 0.008602 | 0.008602 | 0.006519 | 0.542391 |
| stability_penalty | -0.008694 | 0.008694 | 1.000000 | -0.548184 |
| total_reward | 0.010921 | 0.018596 | 1.000000 | 0.688593 |
| generated_reward | 0.010921 | 0.018596 | 1.000000 | 0.688593 |
| original_env_reward | -1.477784 | 2.441471 | 1.000000 | -93.181141 |

## Distribution
- score: mean=-98.037542, min=-118.035399, max=-62.408080
- episode_length: mean=72.300000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
| 2 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -98.04 | -98.04 | 0.00 | 72.30 | distance_penalty=-0.005 progress_delta_reward=0.016 soft_landing_bonus=0.009 stability_penalty=-0.009 | new_best |
