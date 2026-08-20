# ⚠️ 上一版代码验证失败
错误信息：Reward v5 failed validation: runs\env_001\exp_v6\seed_6\iter_05\generation\validations\reward_v5.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -107.157448）
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

    # ---------- 3. soft landing proxy (CONTINUOUS) ----------
    # DIAGNOSIS: binary trigger at 0.1 with 62% nonzero_rate dominates progress 23x.
    # Agent exploits loose binary by hovering, not landing precisely.
    # FIX: replace binary if/else with continuous product of bounded factors.
    # Each factor uses 1/(1+kx) — smooth, bounded [0,1], gradient everywhere.
    # Coefficient 0.15 at max; typical hovering ~0.01-0.02, ratio ~3-7x progress.

    # distance: 1/(1+10*dist) — 0.5 at dist=0.1, 0.09 at dist=1.0
    dist_factor = 1.0 / (1.0 + 10.0 * dist_next)

    # speed: 1/(1+5*(|vx|+|vy|)) — 0.5 at total_speed=0.2, 0.17 at 1.0
    speed_factor = 1.0 / (1.0 + 5.0 * (abs(vx) + abs(vy)))

    # angle: 1/(1+20*|angle|) — 0.5 at |angle|=0.05, 0.33 at 0.1
    angle_factor = 1.0 / (1.0 + 20.0 * abs(angle))

    # contact: soft factor — 0.5 with no contact, 1.0 with both legs
    contact_factor = 0.5 + 0.25 * (left_contact + right_contact)

    soft_landing_proxy = 0.15 * dist_factor * speed_factor * angle_factor * contact_factor

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
score=-107.157448, len=69.900000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta_reward | 0.016190 | 0.017119 | 0.999994 | 1.000000 |
| soft_landing_proxy | 0.001381 | 0.001381 | 1.000000 | 0.085321 |
| stability_penalty | -0.001489 | 0.001489 | 1.000000 | -0.091998 |
| total_reward | 0.016082 | 0.016969 | 1.000000 | 0.993322 |
| generated_reward | 0.016082 | 0.016969 | 1.000000 | 0.993322 |
| original_env_reward | -1.549250 | 2.439284 | 1.000000 | -95.692719 |

## Distribution
- score: mean=-107.157448, min=-122.475523, max=-91.850620
- episode_length: mean=69.900000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta_reward + soft_landing_proxy + stability_penalty | -110.96 | -110.96 | 0.00 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.000 stability_penalty=-0.015 | new_best |
| 2 | progress_delta_reward + soft_landing_proxy + stability_penalty | 156.45 | 156.45 | 0.00 | 916.20 | progress_delta_reward=0.003 soft_landing_proxy=0.061 stability_penalty=-0.001 | new_best |
| 3 | progress_delta_reward + soft_landing_proxy + stability_penalty | 129.76 | 156.45 | -26.69 | 829.80 | progress_delta_reward=0.003 soft_landing_proxy=0.062 stability_penalty=-0.001 | no_meaningful_improvement |
| 4 | progress_delta_reward + soft_landing_proxy + stability_penalty | -107.16 | 156.45 | -263.61 | 69.90 | progress_delta_reward=0.016 soft_landing_proxy=0.001 stability_penalty=-0.001 | no_meaningful_improvement |
