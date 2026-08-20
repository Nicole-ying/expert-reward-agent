# Search objective
- target_score: 200.000000
- current_score: -109.400689
- gap_to_target: 309.400689
- target_achievement_ratio: -54.700%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -109.400689）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D vehicle-like trajectory optimization:
    Reach and stabilize on the central target pad with minimal engine use.
    """
    # Previous observation
    px_prev = obs[0]
    py_prev = obs[1]
    prev_distance = (px_prev**2 + py_prev**2)**0.5

    # Next observation
    px = next_obs[0]
    py = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]

    # Distance to target pad center
    next_distance = (px**2 + py**2)**0.5
    # Speed magnitude
    speed = (vx**2 + vy**2)**0.5

    # 1. Main learning signal: progress delta reward
    #    Positive when approaching target, negative when retreating.
    #    Transforms persistent state penalty into improvement signal.
    progress_delta = 3.0 * (prev_distance - next_distance)

    # 2. Stability constraint: light penalty on high speed, large angle, high angular velocity
    #    Helps agent learn to slow down and keep stable attitude near target
    stability_penalty = -0.1 * speed - 0.05 * abs(angle) - 0.05 * abs(ang_vel)

    # 3. Soft approaching proxy: reward getting close and slow simultaneously
    #    Acts as a smoothed "landing" surrogate without contact signals.
    sigma_dist = 0.2
    sigma_speed = 0.3
    nearness = 2.718281828 ** (-(next_distance**2) / (2.0 * sigma_dist**2))
    slowness = 2.718281828 ** (-(speed**2) / (2.0 * sigma_speed**2))
    soft_landing_reward = 1.0 * nearness * slowness

    # Combine components
    total_reward = progress_delta + stability_penalty + soft_landing_reward

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_reward': soft_landing_reward
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-109.400689, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-123.242094, -78.933292]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_penalty | -7.544432 | -66.3% | 66.3% | 100.0% |
| progress_delta | 3.361815 | 29.6% | 30.6% | 100.0% |
| soft_landing_reward | 0.346006 | 3.0% | 3.0% | 96.1% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个2D类车辆轨迹优化任务。一个主体从视口顶部中央附近开始，带有随机初始力。目标是**尽快到达并稳定在中央目标垫上**，同时**尽可能少地使用引擎推力**。智能体需要学会靠近目标、减速、保持稳定姿态并安全着陆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position — 相对于目标垫的水平坐标
- obs[1]: y_position — 相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 身体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑接触标志（0.0 或 1.0）
- obs[7]: right_support_contact — 右支撑接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不施加任何引擎推力
- action 1: left_orientation_engine — 启动左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 启动主引擎（产生主要推力）
- action 3: right_orientation_engine — 启动右侧姿态引擎（产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
环境在以下任一条件成立时终止（terminated = True）：
- `crash_or_body_contact` — 发生撞击或非安全接触
- `horizontal_position_outside_viewport` — 水平坐标超出视口范围
- `body_not_awake_or_settled` — 身体进入休眠状态或完全静止

分析：
- 前两条通常对应失败（撞击、飞出边界）。
- `body_not_awake_or_settled` 含义模糊：当主体在目标垫上成功稳定着陆时，也会触发静止（成功情况）；但主体可能在错误位置停止或卡住，同样触发。因此该终止**不能直接区分成功与失败**。
- 所有终止信息仅合并为一个布尔值 `terminated`，无额外解释字段。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: info 的所有字段（不可用），original_reward，任何未在 observation 中声明的信号

## 7. 可用于奖励函数的信号
以下信号完全从 `obs`、`action`、`next_obs` 中可获取：
- **位置信号**：x_position, y_position（相对于目标垫），可用于度量到目标的距离
- **速度信号**：x_velocity, y_velocity，可用于惩罚过快运动或鼓励减速
- **姿态信号**：body_angle, angular_velocity，可用于鼓励保持稳定角度（例如竖直）
- **接触信号**：left_support_contact, right_support_contact，可反映着陆脚是否着地、是否平稳
- **动作/引擎信号**：action 选择（0~3），可用于惩罚非零动作以节省燃料

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.