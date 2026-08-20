# Search objective
- target_score: 200.000000
- current_score: -107.649056
- gap_to_target: 307.649056
- target_achievement_ratio: -53.825%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -107.649056）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]

    # Euclidean distances to target platform
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = dist_curr - dist_next

    # Stability: penalise large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    # Controlled descent: reward upward acceleration (deceleration against gravity)
    vy_curr = obs[3]
    vy_next = next_obs[3]
    delta_vy = vy_next - vy_curr
    # Only reward deceleration (positive delta_vy), avoid penalising natural downward acceleration
    vertical_accel_reward = 0.1 * max(0.0, delta_vy)

    total_reward = progress + tilt_penalty + vertical_accel_reward

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty,
        'vertical_accel_reward': vertical_accel_reward
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract current and next positions
    x_curr, y_curr = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    body_angle = obs[4]  # current body angle

    # Euclidean distances to the target platform
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5

    # Main learning signal: progress toward the target
    progress = dist_curr - dist_next

    # Stability constraint: penalize large body tilt
    tilt_penalty = -0.1 * abs(body_angle)

    total_reward = progress + tilt_penalty

    components = {
        'distance_progress': progress,
        'tilt_penalty': tilt_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-107.649056, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-124.065380, -88.710032]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| distance_progress | 1.120728 | 84.1% | 87.0% | 100.0% |
| vertical_accel_reward | 0.153280 | 11.5% | 11.5% | 1.8% |
| tilt_penalty | -0.019398 | -1.5% | 1.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个 2D 飞行器从视口顶部中心附近随机受力启动，需要尽快并安全地降落到中央目标平台（target pad）上并保持稳定。任务鼓励以最快的速度到达目标，同时尽量节约主引擎推力，并维持合理的姿态与轻柔的触地。核心目标是到达并稳定在目标平台上。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: 浮点数（实际可能为 float32），其中索引 6、7 为二值标志（0.0 / 1.0）
- obs[0]: x_position — 相对于目标平台的水平距离
- obs[1]: y_position — 相对于目标平台高度的垂直距离
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体方向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿接触标志（1.0 表示接触，0.0 表示未接触）
- obs[7]: right_support_contact — 右支撑腿接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete (4 个动作)
- action 0: no_engine — 不点火，依靠惯性滑行
- action 1: left_orientation_engine — 点火一个方向舵引擎，通常产生逆时针旋转或平移分量
- action 2: main_engine — 点火主引擎，产生向上或沿机体方向的推力
- action 3: right_orientation_engine — 点火另一个方向舵引擎，通常产生顺时针旋转或平移分量

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 身体不再醒来或处于稳定状态（body_not_awake_or_settled），且大概率双腿均已接触平台（left_support_contact==1.0 且 right_support_contact==1.0），位置和速度都接近零。但没有显式的 success 标志。
- **failure-like termination**:
  1. crash_or_body_contact — 机体与其他区域（如地面、山体）发生碰撞；由于合法接触仅有双腿接触平台，任何其他身体接触均视为失败。
  2. horizontal_position_outside_viewport — 水平方向超出视口边界。
- **ambiguous termination**: body_not_awake_or_settled 但缺少双腿接触或位置偏差过大，可能是未成功着陆的静止状态（如漂浮在空中或仅部分接触平台达到力学平衡）。
- **truncation**: 代码中未见时间步截断，但训练时可能存在 max_episode_steps，此处不展开。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回 info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；不能依赖任何隐含的 success / failure 标志。

## 7. 可用于奖励函数的信号
- **position**: 可用 obs[0] (x 相对距离) 和 obs[1] (y 相对高度)，以及对应的 next_obs 值；可以用来衡量接近目标的进度。
- **velocity**: obs[2], obs[3] 的线速度，可用于鼓励低速着陆。
- **orientation**: obs[4] (角度) 和 obs[5] (角速度)，可用于惩罚倾斜或旋转。
- **contact**: obs[6], obs[7] 的左右支撑腿接触标志，可用于判断是否到达平台，并奖励同时接触。
- **action/engine**: 动作索引本身可用于计算引擎使用惩罚，例如对 main_engine (action 2) 或姿态调节引擎 (action 1, 3) 设置负奖励以节约燃料。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | distance_progress + tilt_penalty | -100.51 | -100.51 | 0.00 | 68.55 | distance_progress=0.016 tilt_penalty=-0.002 | new_best |
| 2 | distance_progress + tilt_penalty + velocity_penalty | -110.36 | -100.51 | -9.85 | 68.45 | distance_progress=0.016 tilt_penalty=-0.002 velocity_penalty=-0.092 | no_meaningful_improvement |
| 3 | distance_progress + tilt_penalty + vertical_accel_reward | -107.65 | -100.51 | -7.14 | 68.40 | distance_progress=0.016 tilt_penalty=-0.001 vertical_accel_reward=0.002 | no_meaningful_improvement |
