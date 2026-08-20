# 上一轮奖励函数代码（该轮得分: 129.761934）
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
    vel_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    ang_vel_penalty = abs(ang_vel)

    w_vel = 0.001
    w_angle = 0.005
    w_angvel = 0.001
    stability_penalty = - (w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy ----------
    # DIAGNOSIS: nonzero_rate=61%, ratio_to_progress=24x — agent exploits loose thresholds.
    # FIX: tighten all three thresholds to make this a rarer, more meaningful signal.
    #   dist: 0.20 -> 0.12, speed: 0.10 -> 0.05, angle: 0.05 -> 0.03
    near_target = dist_next < 0.12
    low_speed = abs(vx) < 0.05 and abs(vy) < 0.05
    upright = abs(angle) < 0.03
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

# 历史最佳奖励函数代码（历史最高得分）
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
score=129.761934, len=829.800000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002685 | 0.002854 | 0.999578 | 1.000000 |
| soft_landing_proxy | 0.062194 | 0.062194 | 0.621939 | 23.164424 |
| stability_penalty | -0.000553 | 0.000553 | 1.000000 | -0.206047 |
| total_reward | 0.064326 | 0.064594 | 1.000000 | 23.958377 |
| generated_reward | 0.064326 | 0.064594 | 1.000000 | 23.958377 |
| original_env_reward | 0.046482 | 1.165589 | 1.000000 | 17.312532 |

## Distribution
- score: mean=129.761934, min=-5.243113, max=186.366770
- episode_length: mean=829.800000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -110.96 | -110.96 | 0.00 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.000 stability_penalty=-0.015 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 156.45 | 156.45 | 0.00 | 916.20 | progress_delta_reward=0.003 soft_landing_proxy=0.061 stability_penalty=-0.001 | new_best |
| 3 | progress_delta_reward + soft_landing_proxy + stability_penalty | 129.76 | 156.45 | -26.69 | 829.80 | progress_delta_reward=0.003 soft_landing_proxy=0.062 stability_penalty=-0.001 | no_meaningful_improvement |
