# Search objective
- target_score: 200.000000
- current_score: -113.995984
- gap_to_target: 313.995984
- target_achievement_ratio: -56.998%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -113.995984）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract next observation variables
    x = next_obs[0]      # x relative to pad center
    y = next_obs[1]      # y relative to pad height
    vx = next_obs[2]     # horizontal velocity
    vy = next_obs[3]     # vertical velocity
    angle = next_obs[4]  # body angle
    omega = next_obs[5]  # angular velocity
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Main drive: distance to target (dense negative signal)
    dist_to_target = (x**2 + y**2) ** 0.5
    reward_dist = -2.0 * dist_to_target

    # 2. Stability constraint: penalise high speeds and large angles
    reward_stability = (
        -0.1 * abs(vx) -
        0.1 * abs(vy) -
        0.1 * abs(angle) -
        0.1 * abs(omega)
    )

    # 3. Soft landing proxy: reward simultaneous near-target, low-speed,
    #    upright attitude and both legs contacting the pad.
    prox_dist_thresh = 0.3
    prox_vel_thresh = 0.2
    prox_angle_thresh = 0.1

    condition = (
        dist_to_target < prox_dist_thresh and
        abs(vx) < prox_vel_thresh and
        abs(vy) < prox_vel_thresh and
        abs(angle) < prox_angle_thresh and
        left_contact > 0.5 and
        right_contact > 0.5
    )
    reward_landing = 1.0 if condition else 0.0

    total_reward = reward_dist + reward_stability + reward_landing

    components = {
        "distance_reward": reward_dist,
        "stability_penalty": reward_stability,
        "soft_landing_proxy": reward_landing
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-113.995984, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_reward | -132.945396 | -93.0% | 93.0% | 100.0% |
| stability_penalty | -9.826903 | -6.9% | 6.9% | 100.0% |
| soft_landing_proxy | 0.200000 | 0.1% | 0.1% | 0.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
