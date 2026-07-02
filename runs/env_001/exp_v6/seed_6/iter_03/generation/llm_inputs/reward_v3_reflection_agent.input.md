# 上一轮奖励函数代码（该轮得分: 156.451604）
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
    # DIAGNOSIS: ratio_to_progress was -0.903, penalty nearly cancels progress signal.
    # FIX: reduce all coefficients by 10x to bring ratio well under 0.5.
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001      # was 0.01
    w_angle = 0.005     # was 0.05
    w_angvel = 0.001    # was 0.01
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (unchanged this round) ----------
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
score=156.451604, len=916.200000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002529 | 0.002736 | 0.999713 | 1.000000 |
| soft_landing_proxy | 0.061126 | 0.061126 | 0.611263 | 24.166297 |
| stability_penalty | -0.000559 | 0.000559 | 1.000000 | -0.220901 |
| total_reward | 0.063097 | 0.063418 | 1.000000 | 24.945396 |
| generated_reward | 0.063097 | 0.063418 | 1.000000 | 24.945396 |
| original_env_reward | 0.039995 | 1.263796 | 1.000000 | 15.812157 |

## Distribution
- score: mean=156.451604, min=52.240116, max=199.098675
- episode_length: mean=916.200000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -110.96 | -110.96 | 0.00 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.000 stability_penalty=-0.015 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 156.45 | 156.45 | 0.00 | 916.20 | progress_delta_reward=0.003 soft_landing_proxy=0.061 stability_penalty=-0.001 | new_best |
