# 上一轮奖励函数代码（该轮得分: -110.961396）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    # obs
    x, y = obs[0], obs[1]
    # next_obs
    nx, ny = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 1. main learning signal: progress toward (0,0) ----------
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress_delta_reward = dist_obs - dist_next

    # ---------- 2. stability / smoothness penalty ----------
    # lightweight penalties to encourage gentle, stable landing
    vel_penalty = abs(vx) + abs(vy)          # prefer zero horizontal/vertical speed
    angle_penalty = abs(angle)               # prefer upright orientation
    ang_vel_penalty = abs(ang_vel)           # prefer no rotation

    w_vel = 0.01
    w_angle = 0.05
    w_angvel = 0.01
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (small bonus for likely safe touchdown) ----------
    # Criteria: near target, low speed, upright, both legs in contact
    near_target = dist_next < 0.2
    low_speed = abs(vx) < 0.1 and abs(vy) < 0.1
    upright = abs(angle) < 0.05
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)

    soft_landing_proxy = 0.1 if (near_target and low_speed and upright and both_contact) else 0.0

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    # ---------- components dict ----------
    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-110.961396, len=69.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016150 | 0.017077 | 0.999991 | 1.000000 |
| soft_landing_proxy | 0.000404 | 0.000404 | 0.004040 | 0.025015 |
| stability_penalty | -0.014589 | 0.014589 | 1.000000 | -0.903374 |
| total_reward | 0.001964 | 0.008432 | 1.000000 | 0.121640 |
| generated_reward | 0.001964 | 0.008432 | 1.000000 | 0.121640 |
| original_env_reward | -1.534872 | 2.395598 | 1.000000 | -95.041096 |

## Distribution
- score: mean=-110.961396, min=-123.943225, max=-97.253096
- episode_length: mean=69.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -110.96 | -110.96 | 0.00 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.000 stability_penalty=-0.015 | new_best |
