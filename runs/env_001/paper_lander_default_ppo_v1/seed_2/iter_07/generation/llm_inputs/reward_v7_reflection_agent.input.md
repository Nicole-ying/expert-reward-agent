# Duplicate reward retry
The previous generation duplicated iter 5 (runs/env_001/paper_lander_default_ppo_v1/seed_2/iter_05/generation/reward_v5.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    time_penalty = -0.02

    total_reward = reward_dist + reward_stability + reward_landing + time_penalty

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing,
        "time_penalty": time_penalty
    }

    return float(total_reward), components
```

# Search objective
- target_score: 200.000000
- current_score: 278.052174
- gap_to_target: -78.052174
- target_achievement_ratio: 139.026%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 278.052174）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    time_penalty = -0.02

    total_reward = reward_dist + reward_stability + reward_landing + time_penalty

    reward_components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing,
        "time_penalty": time_penalty
    }

    return float(total_reward), reward_components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next observation variables
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress signal: reward getting closer to target, penalize moving away
    prev_dist = (prev_x**2 + prev_y**2) ** 0.5
    curr_dist = (x**2 + y**2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    # 2. Stability constraint: light penalty on high speeds and large angles
    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    # 3. Dense landing approach guidance: geometric mean of bounded factors
    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    # Geometric mean prevents product collapse while preserving joint requirement
    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    # Contact factor: partial credit without legs, full credit with both
    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    # 4. Time penalty: small fixed per-step cost to incentivize faster completion
    time_penalty = -0.02

    total_reward = reward_dist + reward_stability + reward_landing + time_penalty

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing,
        "time_penalty": time_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=278.052174, len=231.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[252.076552, 313.464974]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 26.169855 | 75.3% | 75.3% | 100.0% |
| time_penalty | -4.626000 | -13.3% | 13.3% | 100.0% |
| distance_progress | 2.750558 | 7.9% | 8.2% | 93.6% |
| stability_penalty | -1.106092 | -3.2% | 3.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从初始位置（接近视口顶部中心）出发，  
**尽可能快地到达并稳定停在中央的目标着陆垫上**，  
同时尽量减少引擎推力消耗。  
需要学习接近目标、降低速度、保持稳定姿态并安全接触垫面。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（连续值，最后两个分量为 0.0 或 1.0）
- obs[0]: x_position — 相对于目标垫中心的水平坐标
- obs[1]: y_position — 相对于垫面高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact — 右支撑腿接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不启动任何引擎，仅靠惯性
- action 1: left_orientation_engine — 启动左侧姿态引擎
- action 2: main_engine — 启动主引擎（向下推力）
- action 3: right_orientation_engine — 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体稳定/沉睡，表明已安全着陆）
- failure-like termination: crash_or_body_contact（机身非支撑部位触碰地面）、horizontal_position_outside_viewport（水平位置超出视口边界）
- ambiguous termination: 三个条件通过 `or` 逻辑连接，没有独立 success/failure 标志
- truncation: 由环境外部步数限制处理，此处 `step` 不返回截断信号（return 的第四个值为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无任何可用字段）
- forbidden_or_uncertain_info_fields: 全部 info 字段均不可用；不能假设存在 success/failure 等键

## 7. 可用于奖励函数的信号
- position: next_obs[0] (x 相对目标), next_obs[1] (y 相对垫高)
- velocity: next_obs[2], next_obs[3]
- orientation: next_obs[4] (机体角度), next_obs[5] (角速度)
- contact: next_obs[6], next_obs[7]（支撑腿是否触垫）
- action/engine: 动作选择，可对引擎使用（action 1,2,3）施加负代价
- 相邻状态变化：如位置/速度/角度的变化量，可计算趋近或平稳性

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_reward + soft_landing_proxy + stability_penalty | -114.00 | -114.00 | 0.00 | 68.40 | distance_reward=-1.943 soft_landing_proxy=0.004 stability_penalty=-0.146 | new_best |
| 2 | distance_progress + soft_landing_proxy + stability_penalty | -109.88 | -109.88 | 0.00 | 68.50 | distance_progress=0.032 soft_landing_proxy=0.005 stability_penalty=-0.140 | new_best |
| 3 | distance_progress + soft_landing_proxy + stability_penalty | -98.51 | -98.51 | 0.00 | 68.75 | distance_progress=0.032 soft_landing_proxy=0.004 stability_penalty=-0.011 | new_best |
| 4 | distance_progress + landing_approach + stability_penalty | 73.07 | 73.07 | 0.00 | 414.05 | distance_progress=0.005 landing_approach=0.208 stability_penalty=-0.003 | new_best |
| 5 | distance_progress + landing_approach + stability_penalty + time_penalty | 278.05 | 278.05 | 0.00 | 231.30 | distance_progress=0.005 landing_approach=0.223 stability_penalty=-0.002 time_penalty=-0.020 | target_solved_new_best |
| 6 | distance_progress + landing_approach + stability_penalty + time_penalty | 278.05 | 278.05 | 0.00 | 231.30 | distance_progress=0.005 landing_approach=0.223 stability_penalty=-0.002 time_penalty=-0.020 | target_solved_no_improvement |
