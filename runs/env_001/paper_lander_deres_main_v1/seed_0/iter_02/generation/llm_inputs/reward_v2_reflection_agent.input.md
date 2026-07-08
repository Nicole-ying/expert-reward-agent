# 上一轮奖励函数代码（该轮得分: -107.359745）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next states
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_x_vel = next_obs[2]
    next_y_vel = next_obs[3]
    next_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Core learning signal: progress towards target (0,0)
    dist_current = (x ** 2 + y ** 2) ** 0.5
    dist_next = (next_x ** 2 + next_y ** 2) ** 0.5
    progress = dist_current - dist_next  # positive when approaching

    # 2. Soft proxy for landing quality (encourages gentle, two‑legged touchdown)
    contact_term = left_contact * right_contact                     # both feet must touch
    pos_x_score = max(0.0, 1.0 - abs(next_x) / 1.0)
    pos_y_score = max(0.0, 1.0 - abs(next_y) / 1.0)
    pos_score = pos_x_score * pos_y_score
    vel_x_score = max(0.0, 1.0 - abs(next_x_vel) / 1.0)
    vel_y_score = max(0.0, 1.0 - abs(next_y_vel) / 1.0)
    vel_score = vel_x_score * vel_y_score
    angle_score = max(0.0, 1.0 - abs(next_angle) / 1.5)
    landing_quality = contact_term * pos_score * vel_score * angle_score  # [0, 1]

    # 3. Orientation penalty (gentle, keeps the craft upright)
    attitude_penalty = -0.01 * abs(next_angle)

    # Combine
    w_landing = 0.2
    total_reward = progress + w_landing * landing_quality + attitude_penalty

    components = {
        "progress_reward": progress,
        "landing_quality_reward": landing_quality,
        "attitude_penalty": attitude_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-107.359745, len=68.500000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| attitude_penalty | -0.000249 | 0.000249 | 1.000000 | -0.000249 | -0.014625 |
| landing_quality_reward | 0.004575 | 0.004575 | 0.008454 | 0.541194 | 0.269154 |
| progress_reward | 0.016064 | 0.016999 | 0.999991 | 0.016064 | 0.944972 |
| total_reward | 0.016730 | 0.017627 | 1.000000 | 0.016730 | 0.984178 |
| original_env_reward | -1.552170 | 2.413084 | 1.000000 | -1.552170 | -91.308725 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| attitude_penalty | -0.017459 | 0.017459 | -0.931127 | -0.001217 | 14289 |
| landing_quality_reward | 0.321330 | 0.321330 | 0.000000 | 0.956869 | 14289 |
| progress_reward | 1.127941 | 1.127963 | -0.157786 | 1.412644 | 14289 |
| total_reward | 1.174749 | 1.175083 | -0.655217 | 1.564718 | 14289 |

## Distribution
- score: mean=-107.359745, min=-123.927336, max=-78.933292
- episode_length: mean=68.500000
- early_terminal (<150 steps + score<-50): 20/20 (100%)
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
