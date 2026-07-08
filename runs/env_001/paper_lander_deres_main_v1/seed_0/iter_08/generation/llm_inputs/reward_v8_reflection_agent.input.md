# ⚠️ 上一版代码验证失败
错误信息：Reward v8 failed validation: runs\env_001\paper_lander_deres_main_v1\seed_0\iter_08\generation\validations\reward_v8.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -4345.464756）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for lunar-lander-style task (navigation_goal_reaching).
    
    Uses a potential-based shaping as the main progress signal and a soft
    landing proxy as a sparse success approximation.
    """
    
    # ----- Unpack observations -------------------------------------------------
    # Current
    x = obs[0]
    y = obs[1]
    vx = obs[2]
    vy = obs[3]
    angle = obs[4]
    left_contact = obs[6]
    right_contact = obs[7]

    # Next
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_vx = next_obs[2]
    next_vy = next_obs[3]
    next_angle = next_obs[4]
    next_left = next_obs[6]
    next_right = next_obs[7]

    # ----- Derived quantities ---------------------------------------------------
    # Euclidean distance to target (origin)
    dist = (x ** 2.0 + y ** 2.0) ** 0.5
    next_dist = (next_x ** 2.0 + next_y ** 2.0) ** 0.5

    # Speed
    speed = (vx ** 2.0 + vy ** 2.0) ** 0.5
    next_speed = (next_vx ** 2.0 + next_vy ** 2.0) ** 0.5

    # ----- Component 1: Progress reward (potential-based shaping) ----------------
    # Phi(state) = -(dist + β_speed * speed + β_angle * |angle|)
    # The agent receives reward for decreasing distance, slowing down, and
    # aligning upright.
    beta_speed = 0.5
    beta_angle = 0.3
    gamma = 0.99

    phi = -(dist + beta_speed * speed + beta_angle * abs(angle))
    phi_next = -(next_dist + beta_speed * next_speed + beta_angle * abs(next_angle))

    progress_reward = gamma * phi_next - phi

    # ----- Component 2: Landing soft proxy (first touchdown) --------------------
    # Give a one-time bonus when both legs transition from no-contact to contact
    # while velocity and attitude are small (indicating a safe landing).
    landing_reward = 0.0
    landing_bonus = 10.0
    speed_threshold = 0.3
    angle_threshold = 0.1

    both_prev_air = (left_contact == 0.0) and (right_contact == 0.0)
    both_next_ground = (next_left == 1.0) and (next_right == 1.0)
    stable_touchdown = (next_speed < speed_threshold) and (abs(next_angle) < angle_threshold)

    if both_prev_air and both_next_ground and stable_touchdown:
        landing_reward = landing_bonus

    # ----- Total reward ---------------------------------------------------------
    total_reward = progress_reward + landing_reward

    # Components dict (only the terms that are summed)
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Continuous approach quality shaping
    #    Three bounded factors [0,1], product ensures all must be satisfied
    prox_factor = 1.0 / (1.0 + 5.0 * dist_next)
    speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5
    speed_factor = 1.0 / (1.0 + speed)
    angle_factor = 1.0 / (1.0 + 5.0 * abs(next_angle))
    approach_quality = prox_factor * speed_factor * angle_factor  # [0, 1], dense

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # Combine — w_approach reduced from 0.3 to 0.15 to curb hovering
    # (ratio was 280:1 vs progress, now targeting ~140:1 as first step)
    w_approach = 0.15
    approach_quality_reward = w_approach * approach_quality
    total_reward = progress + approach_quality_reward + attitude_penalty

    components = {
        "progress_reward": progress,
        "approach_quality_reward": approach_quality_reward,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-4345.464756, len=383.100000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_reward | 0.000120 | 0.000120 | 0.000012 | 10.000000 | 0.000831 |
| progress_reward | 0.143181 | 0.143849 | 1.000000 | 0.143181 | 0.995353 |
| total_reward | 0.143301 | 0.143969 | 1.000000 | 0.143301 | 0.996184 |
| original_env_reward | -5.786956 | 6.159514 | 1.000000 | -5.786956 | -40.229274 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_reward | 0.083102 | 0.083102 | 0.000000 | 80.000000 | 1444 |
| progress_reward | 99.159312 | 99.159312 | 0.744308 | 330.170053 | 1444 |
| total_reward | 99.242415 | 99.242415 | 0.744308 | 330.170053 | 1444 |

## Distribution
- score: mean=-4345.464756, min=-10799.475542, max=-395.312781
- episode_length: mean=383.100000
- early_terminal (<150 steps + score<-50): 6/20 (30%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化任务。飞行器从视口顶部中央附近以随机初始速度出发，目标是尽可能快地到达并平稳降落在视口中央的指定着陆平台上，同时尽量少用引擎推力。智能体需要学会靠近目标点、降低速度、保持稳定姿态并实现安全接触（着陆腿与平台接触且最终静止）。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（连续值）
- obs[0]: x_position — 机体相对于目标平台中心的水平坐标
- obs[1]: y_position — 机体相对于平台高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑腿是否接触（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右支撑腿是否接触（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无引擎（不做任何推进）
- action 1: 左姿态引擎（触发左侧姿态调整引擎）
- action 2: 主引擎（触发主推进引擎）
- action 3: 右姿态引擎（触发右侧姿态调整引擎）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - `body_not_awake_or_settled`：表示机体已经停止运动（低速、可能双脚着地且稳定），这通常是成功着陆的自然结果。

- failure-like termination:
  - `crash_or_body_contact`：机体与地面或障碍发生非允许的碰撞（如舱体直接撞击地面、侧翻等）。
  - `horizontal_position_outside_viewport`：机体水平位置超出边界，失去控制。

- ambiguous termination:
  - 无。

- truncation:
  - 环境未定义 episode 长度上限（通过其他方式截断），但通常会有一个最大步数限制作为安全截断，此处未明确给出。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无可用字段，step 返回 info = {}）
- forbidden_or_uncertain_info_fields: 任何可能存在的 info 字段均不可用（如 success, failure, termination_reason 等）

## 7. 可用于奖励函数的信号
基于观测空间，可直接使用的信号包括：
- position: obs[0] 横向偏差、obs[1] 垂直偏差
- velocity: obs[2] 水平速度、obs[3] 垂直速度
- orientation: obs[4] 机体倾角、obs[5] 角速度
- contact: obs[6] 左腿接触标志、obs[7] 右腿接触标志
- action/engine: 动作 id（0-3）可用于惩罚引擎使用或鼓励特定策略
- 组合衍生信号：如是否双脚同时接地、速度是否接近零、位置是否接近 0（目标点）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | attitude_penalty + landing_quality_reward + progress_reward | -107.36 | -107.36 | 0.00 | 68.50 | attitude_penalty=-0.000 landing_quality_reward=0.005 progress_reward=0.016 | new_best |
| 2 | approach_quality_reward + attitude_penalty + progress_reward | 124.32 | 124.32 | 0.00 | 873.90 | approach_quality_reward=0.643 attitude_penalty=-0.000 progress_reward=0.002 | new_best |
| 3 | approach_quality_reward + attitude_penalty + progress_reward | 146.36 | 146.36 | 0.00 | 1000.00 | approach_quality_reward=0.094 attitude_penalty=-0.001 progress_reward=0.002 | new_best |
| 4 | approach_quality_reward + attitude_penalty + contact_bonus + progress_reward | 128.65 | 146.36 | -17.71 | 910.25 | approach_quality_reward=0.093 attitude_penalty=-0.000 contact_bonus=0.056 progress_reward=0.002 | no_meaningful_improvement |
| 5 | approach_quality_reward + attitude_penalty + landing_proxy + progress_reward | 139.74 | 146.36 | -6.62 | 1000.00 | approach_quality_reward=0.089 attitude_penalty=-0.001 landing_proxy=0.155 progress_reward=0.002 | no_meaningful_improvement |
| 6 | approach_quality_reward + attitude_penalty + progress_reward | 139.03 | 146.36 | -7.33 | 1000.00 | approach_quality_reward=0.063 attitude_penalty=-0.000 progress_reward=0.002 | unsolved_stagnation_fresh_restart |
| 7 | landing_reward + progress_reward | -4345.46 | 146.36 | -4491.83 | 383.10 | landing_reward=0.000 progress_reward=0.143 | no_meaningful_improvement |
