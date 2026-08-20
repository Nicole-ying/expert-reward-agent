# Search objective
- target_score: 200.000000
- current_score: 214.808066
- gap_to_target: -14.808066
- target_achievement_ratio: 107.404%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 214.808066）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next state variables
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to landing pad center (Euclidean)
    dist = (x**2 + y**2) ** 0.5
    next_dist = (next_x**2 + next_y**2) ** 0.5

    # 1. Main learning signal: progress toward the pad (scaled up)
    w_progress = 5.0
    progress_reward = w_progress * (dist - next_dist)

    # 2. Stability penalty: penalize high velocities
    w_vel = 0.1
    velocity_penalty = -w_vel * (vx**2 + vy**2)

    # 3. Attitude penalty: penalize non-zero body angle (want upright)
    w_angle = 0.01
    attitude_penalty = -w_angle * abs(angle)

    # 4. Soft landing proxy: encourage gentle touchdown with both legs
    w_landing = 2.0
    alpha = 20.0   # sharpness for position proximity
    beta = 5.0     # sharpness for low speed
    contact_factor = left_contact * right_contact  # 1 if both legs touch, else 0
    position_proximity = 2.718281828 ** (-alpha * (next_x**2 + next_y**2))
    speed_term = 2.718281828 ** (-beta * (vx**2 + vy**2))
    landing_proxy = w_landing * contact_factor * position_proximity * speed_term

    total_reward = progress_reward + velocity_penalty + attitude_penalty + landing_proxy

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "attitude_penalty": attitude_penalty,
        "landing_proxy": landing_proxy
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=214.808066, len=557.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[180.151594, 265.668850]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_proxy | 72.100126 | 87.5% | 87.5% | 6.6% |
| progress_reward | 6.906196 | 8.4% | 8.8% | 97.4% |
| velocity_penalty | -2.848134 | -3.5% | 3.5% | 97.5% |
| attitude_penalty | -0.241169 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 车辆型轨迹优化任务。主体从视口顶部中央附近以随机初始力开始运动。目标是尽快到达并稳定停靠在中央的着陆垫上，同时尽可能少地使用引擎推力。智能体需要学会向目标接近、降低速度、维持稳定姿态并安全触垫。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测，连续观测）
- obs[0]: x_position — 相对于着陆垫的水平坐标
- obs[1]: y_position — 相对于垫面高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑触地标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右侧支撑触地标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- action 0: no_engine — 不执行任何推力
- action 1: left_orientation_engine — 点燃一个姿态发动机（产生旋转推力）
- action 2: main_engine — 点燃主发动机（产生向上的推力，通常用于减速/悬停）
- action 3: right_orientation_engine — 点燃另一个姿态发动机（产生反向旋转推力）

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 当机体成功稳定在着陆垫上时（可能体现为 `body_not_awake_or_settled` 且双支撑足接触，且位置靠近垫中心），视为成功着陆。
- failure-like termination: 发生坠毁或身体其他部位接触地面（`crash_or_body_contact`）、水平位置超出视口边界（`horizontal_position_outside_viewport`）均属于失败终止。
- ambiguous termination: 机体在垫外静止或倾斜不稳但仍未触发 crash/越界，此时由 `body_not_awake_or_settled` 触发终止，需结合位置和接触状态判断是否成功，但环境未提供显式成功/失败标志。
- truncation: 未显示时间截断信息，当前 step 源未提及 `truncated` 返回，仅返回 `False`。因此该环境不因步数截断，只可能由上述物理条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 `{}`，未提供任何额外字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为未定义）。不可使用 `info['success']`、`info['failure']` 等不存在字段。

## 7. 可用于奖励函数的信号
- position: `next_obs[0]`（距垫水平距离），`next_obs[1]`（距垫垂直高度）
- velocity: `next_obs[2]`（水平速度），`next_obs[3]`（垂直速度）
- orientation: `next_obs[4]`（姿态角），`next_obs[5]`（角速度）
- contact: `next_obs[6]`（左腿触地），`next_obs[7]`（右腿触地）
- action/engine: `action` 值可用来惩罚引擎使用（如 main engine 点火的能耗）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | attitude_penalty + landing_proxy + progress_reward + velocity_penalty | 93.87 | 93.87 | 0.00 | 878.00 | attitude_penalty=-0.001 landing_proxy=0.846 progress_reward=0.003 velocity_penalty=-0.012 | new_best |
| 2 | attitude_penalty + landing_proxy + progress_reward + velocity_penalty | 214.81 | 214.81 | 0.00 | 557.40 | attitude_penalty=-0.001 landing_proxy=0.875 progress_reward=0.015 velocity_penalty=-0.013 | target_solved_new_best |
