# 1. Search objective
- target_score: 200.000000
- current_score: -68.154281
- gap_to_target: 268.154281

# 2. Current reward program (score: -68.154281)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next state
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Descent encouragement: reward moving downward ----
    descent = max(0.0, obs[1] - next_obs[1])  # positive when going down
    descent_reward = descent * 5.0

    # ---- Proximity penalty: discourage horizontal drift ----
    proximity_penalty = -0.1 * abs(x)

    # ---- Contact stability reward: only when touching the platform ----
    contact = (left_contact + right_contact) > 0.5
    if contact:
        # max 10.0, penalized by deviations from upright, still, and zero angle
        raw_stability = 10.0 - 10.0 * body_angle**2 - 2.0 * vx**2 - 2.0 * vy**2 - 2.0 * angvel**2
        stability_reward = max(0.0, raw_stability)
    else:
        stability_reward = 0.0

    # ---- Fuel penalty: discourage unnecessary engine use ----
    fuel_penalty = -0.05 if action in [1, 2, 3] else 0.0

    # ---- Small per-step penalty to discourage lingering ----
    time_penalty = -0.01

    total = descent_reward + proximity_penalty + stability_reward + fuel_penalty + time_penalty

    components = {
        "descent_reward": descent_reward,
        "proximity_penalty": proximity_penalty,
        "stability_reward": stability_reward,
        "fuel_penalty": fuel_penalty,
        "time_penalty": time_penalty
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-68.154281, len=712.800000, terminated=17/20, truncated=3/20, reward_errors=0
score_range=[-224.490992, 157.404283]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_reward | 360.234230 | 81.7% | 81.7% | 5.5% |
| fuel_penalty | -35.402500 | -8.0% | 8.0% | 99.3% |
| proximity_penalty | -31.232415 | -7.1% | 7.1% | 100.0% |
| time_penalty | -7.128000 | -1.6% | 1.6% | 100.0% |
| descent_reward | 7.122844 | 1.6% | 1.6% | 89.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Component checkpoint trajectories
# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.descent_reward | 0.027453 | 0.027453 | 0.711260 | 0.038598 | 0.038598 | 0.000000 | 0.216222 | 1003520 |
| component.fuel_penalty | -0.038641 | 0.038641 | 0.772819 | -0.050000 | 0.050000 | -0.050000 | 0.000000 | 1003520 |
| component.proximity_penalty | -0.026356 | 0.026356 | 1.000000 | -0.026356 | 0.026356 | -0.101751 | -0.000000 | 1003520 |
| component.stability_reward | 4.688126 | 4.688126 | 0.487009 | 9.626370 | 9.626370 | 0.000000 | 10.000000 | 1003520 |
| component.time_penalty | -0.010000 | 0.010000 | 1.000000 | -0.010000 | 0.010000 | -0.010000 | -0.010000 | 1003520 |
| component.total_reward | 4.640583 | 4.674002 | 1.000000 | 4.640583 | 4.674002 | -0.160576 | 10.117787 | 1003520 |
| generated_reward | 4.640583 | 4.674002 | 1.000000 | 4.640583 | 4.674002 | -0.160576 | 10.117787 | 1003520 |
| original_env_reward | -0.137437 | 1.931610 | 1.000000 | -0.137437 | 1.931610 | -100.000000 | 137.476797 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| descent_reward | 7.332130 | 7.332130 | 3.206240 | 10.610116 | 3754 |
| fuel_penalty | -10.319366 | 10.319366 | -44.650000 | -1.500000 | 3754 |
| proximity_penalty | -7.039178 | 7.039178 | -74.486583 | -0.011121 | 3754 |
| stability_reward | 1252.627098 | 1252.627098 | 0.000000 | 8489.607074 | 3754 |
| time_penalty | -2.670807 | 2.670807 | -10.000000 | -0.540000 | 3754 |
| total_reward | 1239.929877 | 1240.263412 | -23.131855 | 8422.899608 | 3754 |


# 4. Environment facts
## 1. 任务目标
主目标：控制一个从画面顶部中央附近出发的飞行器，安全、稳定地降落在画面中央的目标平台上。要求着陆时速度接近于零、姿态接近竖直，且所有支脚平稳接触平台。

次要目标：在确保主目标达成的前提下，尽量缩短飞行时间，并尽量减少主引擎和姿态引擎的使用（即节省燃料）。

不可混淆的目标：不应将“快速到达”或“节省燃料”凌驾于“安全着陆”之上；也不能将“悬停”或“保持在目标上方”当作成功条件。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- obs[0]: x_position (相对目标平台的水平坐标), 可直接用于距离/接近奖励，reward_usable: true
- obs[1]: y_position (相对目标平台高度的垂直坐标), 同上，reward_usable: true
- obs[2]: x_velocity (水平线速度), 可用于着陆软度控制，reward_usable: true
- obs[3]: y_velocity (垂直线速度), 同上，reward_usable: true
- obs[4]: body_angle (机体朝向角), 可用于姿态奖励，reward_usable: true
- obs[5]: angular_velocity (角速度), 可用于姿态稳定性惩罚，reward_usable: true
- obs[6]: left_support_contact (左侧支脚接触标志，1.0 表示接触), 可用于着陆状态判断，reward_usable: true
- obs[7]: right_support_contact (右侧支脚接触标志，1.0 表示接触), 同上，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力，仅受重力/物理影响)
- action 1: left_orientation_engine (点燃左侧姿态推进器，产生旋转力矩)
- action 2: main_engine (点燃主推进器，提供向上推力并可能产生力矩)
- action 3: right_orientation_engine (点燃右侧姿态推进器，产生反方向旋转力矩)

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: body_not_awake_or_settled 如果发生在飞行器已接触地面且速度/角速度极低时，极可能意味着成功着陆；但如果发生在半空中或刚碰撞后，则可能是早期终止。
- failure‑like termination: crash_or_body_contact（与地面或障碍的异常碰撞）、horizontal_position_outside_viewport（水平飞出边界）明确为失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分成功/失败，需要结合观察判断。
- truncation: 代码中未出现 episode length 截断，但实际部署时可能通过外部 wrapper 实现，当前源中未见。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {} 或未返回任何键）
- forbidden_or_uncertain_info_fields: 任何未在以上列出的字段均不可用

## 7. 可用于奖励函数的信号
- position: x_position, y_position（可直接计算到目标的距离、高度差）
- velocity: x_velocity, y_velocity（可衡量接近速度、着陆软度）
- orientation: body_angle（偏离竖直的角度），angular_velocity（旋转速度）
- contact: left_support_contact, right_support_contact（着陆脚是否触地，可判断着陆状态）
- action/engine: action 索引可映射到是否使用主引擎、姿态引擎，用于推力/燃料惩罚
- other: 可通过 (obs, next_obs) 的组合构造微分信号，如速度变化、角速度变化等