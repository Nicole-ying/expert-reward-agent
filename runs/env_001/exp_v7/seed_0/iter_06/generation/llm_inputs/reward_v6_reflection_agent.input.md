# 上一轮奖励函数代码（该轮得分: 216.838994）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ======== 提取观测 ========
    x_t, y_t = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    vx_next, vy_next = next_obs[2], next_obs[3]
    angle_next = next_obs[4]
    w_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ======== 1. 主学习信号：进步奖励 (progress_delta) ========
    dist_t = (x_t ** 2 + y_t ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_t - dist_next   # 正值为接近目标
    progress_reward = 0.5 * progress   # 缩放因子，使单步典型值在 0.0~0.1 左右

    # ======== 2. 软着陆近似奖励 (soft_landing_proxy) ========
    # 条件：接近目标、低速、角度小、双腿都已接触
    near_target = dist_next < 0.1
    low_speed = (abs(vx_next) + abs(vy_next)) < 0.1
    stable_angle = abs(angle_next) < 0.1
    both_contact = (left_contact > 0.5) and (right_contact > 0.5)
    
    soft_landing_reward = 0.0
    if near_target and low_speed and stable_angle and both_contact:
        soft_landing_reward = 1.0   # 单步事件型奖励，鼓励稳定着陆

    # ======== 3. 轻量稳定性惩罚 (stability_penalty) ========
    # 小权重抑制高速、大角度、高角速度，但不压制主学习信号
    stability_penalty = -0.002 * (abs(vx_next) + abs(vy_next)) \
                        - 0.002 * abs(angle_next) \
                        - 0.001 * abs(w_next)

    # ======== 总奖励 ========
    total_reward = progress_reward + soft_landing_reward + stability_penalty

    # ======== 组件字典 ========
    components = {
        'progress_reward': progress_reward,
        'soft_landing_reward': soft_landing_reward,
        'stability_penalty': stability_penalty,
        'total_reward': total_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=216.838994, len=527.150000, errors=0

## Component evidence

Column definitions:
- `mean`: per-step average of the component value (signed). Positive=reward, negative=penalty.
- `abs_mean`: per-step average of the absolute value. Measures magnitude regardless of sign.
- `nonzero_rate`: fraction of steps where the component value is non-zero (|value| > 1e-12).
- `abs_contrib_%`: this component's share of total reward magnitude. Computed as `100 * abs_mean_of_this_component / sum(abs_mean_of_all_components)`. All components sum to 100%. Use this to judge relative scale — a penalty with high abs_contrib_% may be dominating the learning signal.

| component | mean | abs_mean | nonzero_rate | abs_contrib_% |
|-----------|------|----------|-------------|--------------|
| progress_reward | 0.002365 | 0.002553 | 0.996653 | 0.090013 |
| soft_landing_reward | 0.335935 | 0.335935 | 0.335935 | 11.846313 |
| stability_penalty | -0.001071 | 0.001071 | 1.000000 | 0.037765 |
| total_reward | 0.337229 | 0.337673 | 1.000000 | 11.907631 |
| generated_reward | 0.337229 | 0.337673 | 1.000000 | 11.907631 |
| original_env_reward | -0.146123 | 1.820868 | 1.000000 | 64.210645 |

## Distribution
- score: mean=216.838994, min=75.633064, max=281.041340
- episode_length: mean=527.150000
- early_terminal (<150 steps + score<-50): 0/20 (0%)
- errors: 0


# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -111.87 | -111.87 | 0.00 | 68.45 | distance_reward=-0.971 soft_landing_proxy=0.028 stability_penalty=-0.026 | new_best |
| 2 | distance_reward + soft_landing_continuous + stability_penalty | -115.29 | -111.87 | -3.42 | 68.40 | distance_reward=-0.969 soft_landing_continuous=0.013 stability_penalty=-0.027 | no_meaningful_improvement |
| 3 | contact_bonus + potential_shaping | -1368.99 | -111.87 | -1257.12 | 177.60 | contact_bonus=0.001 potential_shaping=0.126 | no_meaningful_improvement |
| 4 | distance_reward + soft_landing + stability_penalty | -113.14 | -111.87 | -1.27 | 68.40 | distance_reward=-0.971 soft_landing=0.012 stability_penalty=-0.025 | unsolved_stagnation_fresh_restart |
| 5 | progress_reward + soft_landing_reward + stability_penalty | 216.84 | 216.84 | 0.00 | 527.15 | progress_reward=0.002 soft_landing_reward=0.336 stability_penalty=-0.001 | target_solved_new_best |
