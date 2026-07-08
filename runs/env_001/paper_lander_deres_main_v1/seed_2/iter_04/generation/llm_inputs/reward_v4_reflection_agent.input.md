# ⚠️ 上一版代码验证失败
错误信息：Reward v4 failed validation: runs\env_001\paper_lander_deres_main_v1\seed_2\iter_04\generation\validations\reward_v4.validation.json
请修复以上错误，重新生成完整的奖励函数代码。

# 上一轮奖励函数代码（该轮得分: -153.053812）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v3: 移除 landing_bonus 中的 contact_factor，让接近+低速的梯度在触地前就生效。

    诊断：v2 的 landing_bonus 有 contact_factor（二值），导致 nonzero_rate 仅 4.38%。
    95%+ 的 step 没有正向信号，agent 缺乏"减速接近"的引导，crash 率 70%。

    修改：landing_bonus = bonus_scale * dist_factor * speed_factor（去掉 contact_factor）。
    靠近目标垫且速度低即有奖励，不等腿触地。

    Components:
    1. progress_reward:   -distance_to_landing_pad（密集引导，不变）
    2. orientation_penalty: 小惩罚非零姿态角和角速度（不变）
    3. landing_bonus:       连续乘积 —— 距离近 × 速度低（不再要求触地）
    """
    # ---- unpack observations ----
    x_pos = obs[0]
    y_pos = obs[1]
    body_angle = obs[4]
    angular_vel = obs[5]

    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy — proximity × low speed (no contact gate) ----
    # 去掉了 contact_factor，让 agent 在接近+减速时就收到正向梯度
    dist_thresh = 0.5
    speed_thresh = 1.0
    bonus_scale = 10.0

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)

    landing_bonus = bonus_scale * dist_factor * speed_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    reward_v2: 将 landing_bonus 从二值条件改为连续乘积形式，提供着陆梯度。

    Components:
    1. progress_reward:   -distance_to_landing_pad (密集引导)
    2. orientation_penalty: 小惩罚非零姿态角和角速度
    3. landing_bonus:       连续乘积 proxy —— 距离近 × 速度低 × 腿触地
    """
    # ---- unpack observations ----
    x_pos = obs[0]
    y_pos = obs[1]
    body_angle = obs[4]
    angular_vel = obs[5]

    n_x_pos = next_obs[0]
    n_y_pos = next_obs[1]
    n_x_vel = next_obs[2]
    n_y_vel = next_obs[3]
    n_body_angle = next_obs[4]
    n_angular_vel = next_obs[5]
    n_left_contact = next_obs[6]
    n_right_contact = next_obs[7]

    # ---- 1. progress_reward: negative Euclidean distance to target ----
    distance_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    progress_reward = -distance_current

    # ---- 2. orientation_penalty: encourage upright and still attitude ----
    angle_coeff = 0.05
    angvel_coeff = 0.01
    orientation_penalty = -angle_coeff * abs(body_angle) - angvel_coeff * abs(angular_vel)

    # ---- 3. landing_bonus: continuous proxy for successful landing ----
    # 三个连续因子相乘，每个提供梯度，共同约束"接近 + 低速 + 触地"
    dist_thresh = 0.5      # 距离阈值（放宽，提前给梯度）
    speed_thresh = 1.0     # 速度阈值（放宽，提前给梯度）
    bonus_scale = 10.0     # 最大可能奖励（完美着陆时）

    n_distance = (n_x_pos ** 2 + n_y_pos ** 2) ** 0.5
    n_speed = (n_x_vel ** 2 + n_y_vel ** 2) ** 0.5

    dist_factor = max(0.0, 1.0 - n_distance / dist_thresh)
    speed_factor = max(0.0, 1.0 - n_speed / speed_thresh)
    contact_factor = min(1.0, n_left_contact + n_right_contact)  # 0, 0→0; 1, 0→1; 1, 1→1

    landing_bonus = bonus_scale * dist_factor * speed_factor * contact_factor

    # ---- total reward ----
    total_reward = progress_reward + orientation_penalty + landing_bonus

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "landing_bonus": landing_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Training outcome
score=-153.053812, len=133.900000, errors=0
`score` 是评估回合的外部环境累计奖励均值；下表组件的 `mean` 是按 step 统计的均值，两者不能直接按数值大小比较。

## Component evidence

`ratio_to_progress_reward` = mean_of_component / abs_mean_of_progress_reward. Signed ratio relative to the main learning signal. Positive = same direction, negative = opposite direction. All components are expressed in units of the main signal.

| component | mean | abs_mean | nonzero_rate | mean_when_active | ratio_to_progress_reward |
|-----------|------|----------|-------------|------------------|--------------------------|
| landing_bonus | 1.212373 | 1.212373 | 0.253987 | 4.773368 | 1.548481 |
| orientation_penalty | -0.018956 | 0.018956 | 1.000000 | -0.018956 | -0.024211 |
| progress_reward | -0.782943 | 0.782943 | 1.000000 | -0.782943 | -1.000000 |
| total_reward | 0.410474 | 1.909171 | 1.000000 | 0.410474 | 0.524270 |
| original_env_reward | -1.591747 | 3.602428 | 1.000000 | -1.591747 | -2.033030 |

> `ratio_to_progress_reward` 把所有组件归一化到主学习信号的尺度。正值=同向，负值=反向。`original_env_reward` 仅用于对齐参考——不参与训练。如果它的 ratio 符号与主信号相反，奖励函数可能 misaligned。

## Per-episode component contribution

| component | episode_sum_mean | episode_sum_abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| landing_bonus | 119.283934 | 119.283934 | 0.000000 | 8001.381261 | 10190 |
| orientation_penalty | -1.865463 | 1.865463 | -18.949881 | -0.048128 | 10190 |
| progress_reward | -77.057154 | 77.057154 | -574.683072 | -41.965808 | 10190 |
| total_reward | 40.361317 | 161.360132 | -460.289887 | 7905.675253 | 10190 |

## Distribution
- score: mean=-153.053812, min=-249.691188, max=-32.767124
- episode_length: mean=133.900000
- early_terminal (<150 steps + score<-50): 17/20 (85%)
- errors: 0


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
一个 2D 飞行器从画面顶部中央出发，受随机初始力影响。核心目标是让飞行器尽快到达画面中央的目标着陆垫并稳定停稳，同时尽量减少引擎推力使用。智能体需要学习靠近目标、减速、保持姿态稳定并实现安全着陆。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: 飞行器相对于目标垫中心的水平坐标 (x_position)
- obs[1]: 飞行器相对于目标垫高度的垂直坐标 (y_position)
- obs[2]: 水平线速度 (x_velocity)
- obs[3]: 垂直线速度 (y_velocity)
- obs[4]: 机身姿态角 (body_angle)
- obs[5]: 角速度 (angular_velocity)
- obs[6]: 左支撑腿接触标志，接触为 1.0，否则为 0.0 (left_support_contact)
- obs[7]: 右支撑腿接触标志，接触为 1.0，否则为 0.0 (right_support_contact)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: 无引擎推力 (no_engine)
- action 1: 点燃左侧姿态引擎 (left_orientation_engine)
- action 2: 点燃主引擎 (main_engine)
- action 3: 点燃右侧姿态引擎 (right_orientation_engine)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（飞行器停止活动并稳定，可能表示成功着陆并停稳）
- failure-like termination: crash_or_body_contact（撞击或非正常身体接触），horizontal_position_outside_viewport（水平出界）
- ambiguous termination: body_not_awake_or_settled 在未接触目标垫或错误位置时可能不表示真正成功
- truncation: 未明确提及，假设无时间截断

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，info为空字典）
- forbidden_or_uncertain_info_fields: info（空字典，不可提供任何额外信号）

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])，可用于衡量与目标垫的距离
- velocity: x_velocity (obs[2]), y_velocity (obs[3])，可用于衡量降落时的速度大小
- orientation: body_angle (obs[4]), angular_velocity (obs[5])，可用于姿态稳定
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])，可用于判断着陆
- action/engine: 动作选择本身（主引擎或姿态引擎），可用于惩罚推力使用

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | landing_bonus + orientation_penalty + progress_reward | -161.08 | -161.08 | 0.00 | 77.85 | landing_bonus=0.254 orientation_penalty=-0.020 progress_reward=-0.945 | new_best |
| 2 | landing_bonus + orientation_penalty + progress_reward | -120.87 | -120.87 | 0.00 | 77.85 | landing_bonus=0.183 orientation_penalty=-0.020 progress_reward=-0.935 | new_best |
| 3 | landing_bonus + orientation_penalty + progress_reward | -153.05 | -120.87 | -32.18 | 133.90 | landing_bonus=1.212 orientation_penalty=-0.019 progress_reward=-0.783 | no_meaningful_improvement |
