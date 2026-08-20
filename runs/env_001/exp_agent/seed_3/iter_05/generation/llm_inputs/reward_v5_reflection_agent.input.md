# 上一轮奖励函数代码（该轮得分: -111.552558）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Helper: distance to goal ==========
    def dist_to_goal(x, y):
        return (x**2 + y**2) ** 0.5

    # ========== 1. Progress delta reward (main learning signal) ==========
    d_current = dist_to_goal(obs[0], obs[1])
    d_next = dist_to_goal(next_obs[0], next_obs[1])
    progress_delta = d_current - d_next   # positive when getting closer

    # ========== 2. Stability penalty with distance gating ==========
    # Unchanged — well-balanced (ratio ~ -1.6)
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: coefficient 0.4 → 0.25) ==========
    # Rationale: iter 3 showed continuous product works (score +32%), but ratio 47.8 means
    # landing proxy dominates total reward (85% contribution). Reducing coefficient to 0.25
    # brings expected ratio from ~48 down to ~30 while preserving gradient signal.

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0                           # 0 / 0.5 / 1.0

    soft_landing_proxy = 0.25 * prox_factor * speed_factor * angle_factor * contact_factor

    # ========== Total reward ==========
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== Helper: distance to goal ==========
    def dist_to_goal(x, y):
        return (x**2 + y**2) ** 0.5

    # ========== 1. Progress delta reward (main learning signal) ==========
    d_current = dist_to_goal(obs[0], obs[1])
    d_next = dist_to_goal(next_obs[0], next_obs[1])
    progress_delta = d_current - d_next   # positive when getting closer

    # ========== 2. Stability penalty with distance gating ==========
    # Unchanged from last iteration — already well-balanced (ratio ~ -1.06)
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: binary → continuous product) ==========
    # Rationale: binary if-condition gives no gradient. Replaced with product of
    # bounded factors max(0, 1-x/T), each providing gradient toward the desired state.
    # The product naturally enforces "all conditions simultaneously" while still
    # giving partial credit when some (but not all) are met.

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0                           # 0 / 0.5 / 1.0

    soft_landing_proxy = 0.4 * prox_factor * speed_factor * angle_factor * contact_factor

    # ========== Total reward ==========
    total_reward = 10.0 * progress_delta + stability_penalty + soft_landing_proxy

    components = {
        "progress_delta": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-111.552558, len=70.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.016147 | 0.017074 | 0.999993 | 1.000000 |
| soft_landing_proxy | 0.000846 | 0.000846 | 0.006256 | 0.052368 |
| stability_penalty | -0.011309 | 0.011309 | 1.000000 | -0.700407 |
| total_reward | 0.151006 | 0.162492 | 1.000000 | 9.351962 |
| generated_reward | 0.151006 | 0.162492 | 1.000000 | 9.351962 |
| original_env_reward | -1.610148 | 2.431917 | 1.000000 | -99.718274 |

## Distribution
- score: mean=-111.552558, min=-124.463377, max=-98.869122
- episode_length: mean=70.600000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_delta + soft_landing_proxy + stability_penalty | -109.27 | -109.27 | 0.00 | 70.60 | progress_delta=0.016 soft_landing_proxy=0.001 stability_penalty=-0.147 | new_best |
| 2 | progress_delta + soft_landing_proxy + stability_penalty | 148.74 | 148.74 | 0.00 | 745.50 | progress_delta=0.011 soft_landing_proxy=0.017 stability_penalty=-0.012 | new_best |
| 3 | progress_delta + soft_landing_proxy + stability_penalty | 197.22 | 197.22 | 0.00 | 796.30 | progress_delta=0.004 soft_landing_proxy=0.192 stability_penalty=-0.006 | new_best |
| 4 | progress_delta + soft_landing_proxy + stability_penalty | -111.55 | 197.22 | -308.77 | 70.60 | progress_delta=0.016 soft_landing_proxy=0.001 stability_penalty=-0.011 | no_meaningful_improvement |
