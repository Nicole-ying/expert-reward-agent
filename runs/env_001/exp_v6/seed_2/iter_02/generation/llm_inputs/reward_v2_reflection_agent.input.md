# ⚠️ 上一版代码验证失败
错误信息：Reward v2 failed validation: runs\env_001\exp_v6\seed_2\iter_02\generation\validations\reward_v2.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -108.614223）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- position and progress delta ----
    px, py = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    d_obs = (px**2 + py**2) ** 0.5
    d_next = (nx**2 + ny**2) ** 0.5
    progress_delta = d_obs - d_next

    # ---- distance penalty (anchor, tiny weight) ----
    distance_penalty = -0.05 * d_next

    # ---- stability penalty (small weight) ----
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    stability_penalty = (
        -0.05 * abs(vx)
        - 0.05 * abs(vy)
        - 0.2 * abs(angle)
        - 0.2 * abs(ang_vel)
    )

    # ---- soft landing proxy (first double-contact landing) ----
    bonus = 0.0
    # Constants – tunable thresholds
    LANDING_DIST_THRESH = 0.5
    VEL_THRESH = 0.2
    ANGLE_THRESH = 0.1

    # New double contact?
    new_contact = (
        next_obs[6] == 1.0 and next_obs[7] == 1.0 and
        (obs[6] != 1.0 or obs[7] != 1.0)
    )
    if new_contact:
        if d_next < LANDING_DIST_THRESH:
            if abs(vx) < VEL_THRESH and abs(vy) < VEL_THRESH:
                if abs(angle) < ANGLE_THRESH:
                    bonus = 1.0

    # ---- total ----
    total = progress_delta + distance_penalty + stability_penalty + bonus

    components = {
        "progress_delta_reward": progress_delta,
        "distance_penalty": distance_penalty,
        "stability_penalty": stability_penalty,
        "soft_landing_bonus": bonus,
        "total_reward": total
    }
    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-108.614223, len=72.000000, errors=0

## Component evidence

| component | mean | abs_mean | nonzero_rate | ratio_to_progress |
|-----------|------|----------|-------------|------------------|
| distance_penalty | -0.048590 | 0.048590 | 1.000000 | -3.023055 |
| progress_delta_reward | 0.016073 | 0.017006 | 0.999993 | 1.000000 |
| soft_landing_bonus | 0.000532 | 0.000532 | 0.000532 | 0.033107 |
| stability_penalty | -0.081839 | 0.081839 | 1.000000 | -5.091688 |
| total_reward | -0.113824 | 0.114840 | 1.000000 | -7.081637 |
| generated_reward | -0.113824 | 0.114840 | 1.000000 | -7.081637 |
| original_env_reward | -1.525328 | 2.369727 | 1.000000 | -94.899531 |

## Distribution
- score: mean=-108.614223, min=-120.796346, max=-95.059093
- episode_length: mean=72.000000
- early_terminal (<150 steps + score<-50): 10/10 (100%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_penalty + progress_delta_reward + soft_landing_bonus + stability_penalty | -108.61 | -108.61 | 0.00 | 72.00 | distance_penalty=-0.049 progress_delta_reward=0.016 soft_landing_bonus=0.001 stability_penalty=-0.082 | new_best |
