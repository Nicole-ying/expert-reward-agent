# ⚠️ 上一版代码验证失败
错误信息：Reward v6 failed validation: runs\env_001\exp_agent\seed_3\iter_06\generation\validations\reward_v6.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -110.742220）
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
    # Unchanged — well-balanced across all iterations
    distance_gate = 1.0 / (1.0 + 5.0 * d_next)
    w_vel   = 0.03
    w_angle = 0.15
    w_omega = 0.03
    speed_penalty = w_vel * (abs(next_obs[2]) + abs(next_obs[3]))
    angle_penalty = w_angle * abs(next_obs[4])
    omega_penalty = w_omega * abs(next_obs[5])
    stability_penalty = -distance_gate * (speed_penalty + angle_penalty + omega_penalty)

    # ========== 3. Continuous soft landing proxy (CHANGED: epsilon on contact_factor) ==========
    # Rationale: the product form prox * speed * angle * contact has zero gradient for ALL
    # factors when contact_factor = 0 (chain rule: derivative involves product of other three).
    # This means the agent gets NO gradient toward the goal until it accidentally touches a leg.
    # Adding epsilon=0.05 to contact_factor ensures gradient always flows through the product,
    # while leg contact still provides a strong boost (0.55 or 1.05 vs baseline 0.05).
    # Coefficient restored to 0.4 (validated in iter 3, score 197).

    prox_factor   = max(0.0, 1.0 - d_next / 0.5)                                # 0→1 as dist 0.5→0
    speed_factor  = max(0.0, 1.0 - (abs(next_obs[2]) + abs(next_obs[3])) / 0.5)  # 0→1 as total speed 0.5→0
    angle_factor  = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)                       # 0→1 as |angle| 0.3→0
    contact_factor = (next_obs[6] + next_obs[7]) / 2.0 + 0.05                    # 0.05 / 0.55 / 1.05

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
score=-110.742220, len=70.600000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| progress_delta | 0.016137 | 0.017054 | 0.999999 | 1.000000 |
| soft_landing_proxy | 0.001456 | 0.001456 | 0.006490 | 0.090250 |
| stability_penalty | -0.011195 | 0.011195 | 1.000000 | -0.693739 |
| total_reward | 0.151631 | 0.162919 | 1.000000 | 9.396511 |
| generated_reward | 0.151631 | 0.162919 | 1.000000 | 9.396511 |
| original_env_reward | -1.602028 | 2.425117 | 1.000000 | -99.276744 |

## Distribution
- score: mean=-110.742220, min=-124.592751, max=-97.569959
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
| 5 | progress_delta + soft_landing_proxy + stability_penalty | -110.74 | 197.22 | -307.96 | 70.60 | progress_delta=0.016 soft_landing_proxy=0.001 stability_penalty=-0.011 | no_meaningful_improvement |
