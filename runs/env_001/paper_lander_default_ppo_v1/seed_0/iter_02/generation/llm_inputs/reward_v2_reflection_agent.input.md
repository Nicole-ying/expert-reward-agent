# Search objective
- target_score: 200.000000
- current_score: -100.497628
- gap_to_target: 300.497628
- target_achievement_ratio: -50.249%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -100.497628）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward for 2D lander: drive the agent to reach and settle on the central pad.
    Components:
      - progress_delta_reward: reward reduction in distance to the target (0,0)
      - stability_penalty: small penalty for high speed, tilt and angular velocity
      - soft_landing_proxy: bonus when near target, slow, upright and both feet in contact
    All components use only allowed signals (obs, next_obs, action). No terminal flags used.
    """
    # -- Helper: distance to target (target is at (0,0))
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next    = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # -- 1. Progress delta (main driving signal)
    delta_dist = dist_current - dist_next
    w_progress = 1.0
    progress_reward = w_progress * delta_dist

    # -- 2. Stability penalty (light constraint) on next state
    w_vel   = 0.01   # penalty per unit speed sum
    w_angle = 0.01   # penalty per radian of body angle
    w_angvel= 0.001  # penalty per angular velocity unit

    stability_penalty = (
        -w_vel   * (abs(next_obs[2]) + abs(next_obs[3]))
        -w_angle * abs(next_obs[4])
        -w_angvel* abs(next_obs[5])
    )

    # -- 3. Soft landing proxy: conditions for being safely on the pad
    dist_threshold   = 0.5
    vel_threshold    = 0.3
    angle_threshold  = 0.2   # rad (~11.5 deg)

    near_target = dist_next < dist_threshold
    low_speed   = (abs(next_obs[2]) < vel_threshold) and (abs(next_obs[3]) < vel_threshold)
    upright     = abs(next_obs[4]) < angle_threshold
    both_contact = (next_obs[6] > 0.5) and (next_obs[7] > 0.5)

    w_proxy = 0.5
    soft_landing_proxy = w_proxy if (near_target and low_speed and upright and both_contact) else 0.0

    # -- Total reward
    total_reward = progress_reward + stability_penalty + soft_landing_proxy

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing_proxy
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-100.497628, len=68.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-125.123549, -58.052384]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 1.118053 | 49.0% | 50.8% | 100.0% |
| stability_penalty | -0.947011 | -41.5% | 41.5% | 100.0% |
| soft_landing_proxy | 0.175000 | 7.7% | 7.7% | 0.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器（类似着陆器）轨迹优化任务。主体从接近画面顶部中心的位置开始，并带有一个随机初始力。任务是**尽快到达画面中央的固定目标平台并稳定停驻**，同时**尽量少使用引擎推力**。智能体需要学会：朝目标接近、减速、保持稳定朝向、安全接触并稳定在平台上。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32
- obs[0]: x_position – 飞行器相对于目标平台中心的水平坐标
- obs[1]: y_position – 飞行器相对于平台高度（pad height）的垂直坐标
- obs[2]: x_velocity – 水平线速度
- obs[3]: y_velocity – 垂直线速度
- obs[4]: body_angle – 机体朝向角度
- obs[5]: angular_velocity – 角速度
- obs[6]: left_support_contact – 左侧支撑脚接触标志（0.0 或 1.0）
- obs[7]: right_support_contact – 右侧支撑脚接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete (4 个动作)
- action 0: no_engine – 不启动任何引擎，仅凭惯性运动
- action 1: left_orientation_engine – 点燃一侧的姿态调整引擎（产生角力矩）
- action 2: main_engine – 点燃主引擎（产生向下的推力）
- action 3: right_orientation_engine – 点燃另一侧的姿态调整引擎（相反方向角力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **body_not_awake_or_settled** 可能表示飞行器稳定停驻在目标平台上（如果位置在目标附近且速度很低），但需要结合观测判断，无显式成功标记。
- failure-like termination: **crash_or_body_contact**（非平台且因撞击导致终止）、**horizontal_position_outside_viewport**（水平飞出边界）都极可能是失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分停在平台上还是停在平台外，需要观测才能确定。
- truncation: 可能由最大步数截断，但源码中未体现，info 为空，无法区分。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 在 step 返回中为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 任何 info 键均不可用（因返回 `{}`）

## 7. 可用于奖励函数的信号
- position: obs[0] x_position（相对目标水平偏移）、obs[1] y_position（相对平台高度）
- velocity: obs[2] x_velocity, obs[3] y_velocity（速度项，可用于惩罚高速或奖励低速靠近）
- orientation: obs[4] body_angle（朝向角度，希望尽量竖直）
- angular velocity: obs[5] angular_velocity（平稳性）
- contact: obs[6] left_support_contact, obs[7] right_support_contact（接触指示，可能用于判断着陆状态）
- action/engine: action 所代表的引擎使用情况（可惩罚使用主引擎或姿态引擎）

# Compact expert route context
- selected_route_id: navigation_goal_reaching
- recommended_design_roles: terminal_success_reward (终点成功奖励), terminal_failure_penalty (失败惩罚), time_penalty (每步时间惩罚), distance_reward (距离型密集奖励), progress_delta_reward (增量进步奖励), potential_based_shaping (势能塑形奖励), gated_reward (门控/层级奖励)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + soft_landing_proxy + stability_penalty | -100.50 | -100.50 | 0.00 | 68.80 | progress_reward=0.016 soft_landing_proxy=0.003 stability_penalty=-0.014 | new_best |
