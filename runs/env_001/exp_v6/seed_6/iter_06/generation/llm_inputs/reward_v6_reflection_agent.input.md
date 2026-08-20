# 上一轮奖励函数代码（该轮得分: 249.016017）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---------- extract states ----------
    x, y = obs[0], obs[1]
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
    stability_penalty = -(w_vel * vel_penalty + w_angle * angle_penalty + w_angvel * ang_vel_penalty)

    # ---------- 3. soft landing proxy (CONTINUOUS, max(0,1-x/D) form) ----------
    # DIAGNOSIS: previous 1/(1+kx) four-factor product collapsed to mean=0.001,
    # too weak to guide. max(0,1-x/D) stays near 1.0 when close to ideal,
    # so product doesn't collapse. Three factors: distance, speed, angle.
    # Coefficient 0.4 gives max 0.4 at perfect landing, ~0.05-0.15 during approach.

    # distance factor: 1.0 at dist=0, 0.0 at dist>=0.5
    dist_factor = max(0.0, 1.0 - dist_next / 0.5)

    # speed factor: 1.0 at zero speed, 0.0 at total_speed>=0.5
    total_speed = abs(vx) + abs(vy)
    speed_factor = max(0.0, 1.0 - total_speed / 0.5)

    # angle factor: 1.0 at angle=0, 0.0 at |angle|>=0.2
    angle_factor = max(0.0, 1.0 - abs(angle) / 0.2)

    # contact: soft factor [0.33, 1.0], rewards any contact
    contact_factor = 0.33 + 0.335 * (left_contact + right_contact)

    soft_landing_proxy = 0.4 * dist_factor * speed_factor * angle_factor * contact_factor

    # ---------- total ----------
    total_reward = progress_delta_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta_reward": progress_delta_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=249.016017, len=276.100000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.002800 | 0.003082 | 0.999744 | 1.000000 |
| soft_landing_proxy | 0.205860 | 0.205860 | 0.697251 | 73.523114 |
| stability_penalty | -0.000707 | 0.000707 | 1.000000 | -0.252563 |
| total_reward | 0.207952 | 0.208327 | 1.000000 | 74.270551 |
| generated_reward | 0.207952 | 0.208327 | 1.000000 | 74.270551 |
| original_env_reward | -0.006390 | 1.578897 | 1.000000 | -2.282318 |

## Distribution
- score: mean=249.016017, min=11.977674, max=305.768457
- episode_length: mean=276.100000
- early_terminal (<150 steps + score<-50): 0/10 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -110.96 | -110.96 | 0.00 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.000 stability_penalty=-0.015 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 156.45 | 156.45 | 0.00 | 916.20 | progress_delta_reward=0.003 soft_landing_proxy=0.061 stability_penalty=-0.001 | new_best |
| 3 | progress_delta_reward + soft_landing_proxy + stability_penalty | 129.76 | 156.45 | -26.69 | 829.80 | progress_delta_reward=0.003 soft_landing_proxy=0.062 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.16 | 156.45 | -263.61 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.001 | no_meaningful_improvement |
| 5 | progress_delta_reward + soft_landing_proxy + stability_penalty | 249.02 | 249.02 | 0.00 | 276.10 | progress_delta_reward=0.003 soft_landing_proxy=0.206 stability_penalty=-0.001 | target_solved_new_best |
