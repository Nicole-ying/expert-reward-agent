# Search objective
- target_score: 200.000000
- current_score: -150.567820
- gap_to_target: 350.567820
- target_achievement_ratio: -75.284%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -150.567820）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -- Extract state variables --
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]
    left_contact_n = next_obs[6]
    right_contact_n = next_obs[7]

    # -- Component 1: Potential-based progress (unchanged) --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty (unchanged) --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Dense landing guidance (replaces sparse product) --
    # Proximity measures that are high when close to the ideal landing state.
    # Each dimension uses max(0, 1 - |error|/threshold) to give a fraction in [0,1].
    x_proximity        = max(0.0, 1.0 - abs(x_n) / 0.4)
    y_proximity        = max(0.0, 1.0 - abs(y_n) / 0.4)
    speed_n            = (vx_n**2 + vy_n**2) ** 0.5
    speed_proximity    = max(0.0, 1.0 - speed_n / 0.3)
    angle_proximity    = max(0.0, 1.0 - abs(angle_n) / 0.2)
    contact_proximity  = (left_contact_n + right_contact_n) / 2.0   # 0, 0.5 or 1

    # Weighted sum gives a dense indicator of landing progress.
    # Weights chosen so that the maximum per-step reward is on the order of 1.0.
    w_x   = 0.15
    w_y   = 0.25
    w_spd = 0.25
    w_ang = 0.25
    w_con = 0.10

    landing_guidance = w_x * x_proximity + w_y * y_proximity + \
                       w_spd * speed_proximity + w_ang * angle_proximity + \
                       w_con * contact_proximity

    k_landing_guidance = 2.0
    landing_reward = k_landing_guidance * landing_guidance

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + landing_reward

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "landing_reward": float(landing_reward)
    }

    return float(total_reward), components
```

# 历史最佳奖励函数代码（历史最高得分）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -- Extract state variables --
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle, angular_v = obs[4], obs[5]

    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n, angular_v_n = next_obs[4], next_obs[5]
    left_contact_n = next_obs[6]
    right_contact_n = next_obs[7]

    # -- Component 1: Potential-based progress --
    w_dist = 1.0
    w_vel  = 0.3
    dist_current = (x**2 + y**2) ** 0.5
    dist_next    = (x_n**2 + y_n**2) ** 0.5
    vel_current  = abs(vx) + abs(vy)
    vel_next     = abs(vx_n) + abs(vy_n)

    potential_current = w_dist * dist_current + w_vel * vel_current
    potential_next    = w_dist * dist_next    + w_vel * vel_next
    potential_diff = potential_current - potential_next

    # -- Component 2: Attitude penalty --
    k_angle = 0.5
    k_angvel = 0.1
    angle_penalty = -k_angle * (angle_n**2) - k_angvel * (angular_v_n**2)

    # -- Component 3: Dense landing quality reward --
    # Provides per-step reward when both legs touch the ground and the pose is near-ideal.
    # All conditions use a linear decay toward zero outside the thresholds.
    x_ok = max(0.0, 1.0 - abs(x_n) / 0.2)
    y_ok = max(0.0, 1.0 - abs(y_n) / 0.3)
    vx_ok = max(0.0, 1.0 - abs(vx_n) / 0.2)
    vy_ok = max(0.0, 1.0 - abs(vy_n) / 0.2)
    angle_ok = max(0.0, 1.0 - abs(angle_n) / 0.2)
    contact_both = left_contact_n * right_contact_n   # 0.0 or 1.0

    landing_quality = contact_both * x_ok * y_ok * vx_ok * vy_ok * angle_ok
    k_landing = 5.0
    landing_quality_reward = k_landing * landing_quality

    # -- Total reward --
    total_reward = potential_diff + angle_penalty + landing_quality_reward

    components = {
        "potential_diff": float(potential_diff),
        "angle_penalty": float(angle_penalty),
        "landing_quality_reward": float(landing_quality_reward)
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-150.567820, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-186.063914, -110.291321]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 1058.594718 | 99.3% | 99.3% | 100.0% |
| potential_diff | 0.064415 | 0.0% | 0.5% | 100.0% |
| angle_penalty | -2.015450 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 2D 飞行器从视口顶部中央附近出发，尽快飞到画面中心的着陆垫上并稳定停靠。过程中需要尽量节省引擎推力，同时避免发生碰撞、水平出界或提前休眠。最终期望的位置是着陆垫上方、接近静止、姿态稳定且左右支撑脚均与垫子接触。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据常规推断，其中 contact 标志用 0.0/1.0 表示）
- obs[0] (x_position): 飞行器相对着陆垫中心的水平坐标（可能是负、零或正）
- obs[1] (y_position): 飞行器相对着陆垫高度的垂直坐标（正值在上方，负值在下方）
- obs[2] (x_velocity): 水平线速度
- obs[3] (y_velocity): 垂直线速度
- obs[4] (body_angle): 机体朝向角度（弧度或度？具体范围由物理实现决定，但在奖励中应处理为接近 0 表示竖直姿态）
- obs[5] (angular_velocity): 角速度
- obs[6] (left_support_contact): 左支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）
- obs[7] (right_support_contact): 右支撑脚是否接触着陆垫（0.0 无接触，1.0 接触）

## 4. 动作空间 action_space
- type: Discrete
- action 0: no_engine —— 什么都不做（不点火，惯性和风力影响下自由飘动）
- action 1: left_orientation_engine —— 点火某个方向的姿态控制引擎（用于向左旋转机体）
- action 2: main_engine —— 点火主引擎（提供向上的推力，用于减速或上升）
- action 3: right_orientation_engine —— 点火相反方向的姿态控制引擎（用于向右旋转机体）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 飞行器在着陆垫上方稳定着陆，速度很小，角度接近 0，且左右支撑脚均接触（即 body_not_awake_or_settled 触发，同时位置、速度、角度、接触均满足期望状态）。该情形隐含“成功到达目标并安全停靠”。
- failure-like termination: 
  - crash_or_body_contact（飞行器主体与地面或边界发生破坏性碰撞）
  - horizontal_position_outside_viewport（飞行器水平飞出画面）
- ambiguous termination: body_not_awake_or_settled 可能既包含成功着陆（满足条件），也可能包含飞行器在非目标位置提前休眠（例如坠落在远处后不动、或能量耗尽停止）。因此该终止条件本身不能直接等同于成功，必须结合位置、接触等信息判断。
- truncation: 源码中为 `False`，未显示任何截断机制，可假设无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无任何显式标记）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空，无有效字段可用）
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空，不能假设存在 success、failure 或其他标记）

## 7. 可用于奖励函数的信号
以下信号均可从 next_obs（或结合 obs 计算增量）中提取，且属于任务相关的可靠特征：

- position: x_position, y_position（obs[0], obs[1]）—— 可衡量距着陆目标的远近、是否处于垫子正上方
- velocity: x_velocity, y_velocity（obs[2], obs[3]）—— 速度大小影响安全着陆和平顺性
- orientation: body_angle（obs[4]）—— 姿态是否竖直（接近 0）
- angular_velocity: obs[5] —— 旋转是否稳定
- contact: left_support_contact, right_support_contact（obs[6], obs[7]）—— 是否双脚接触，判断着陆状态
- action/engine: 可依据动作是否使用主引擎或姿态引擎来鼓励节能（如惩罚主引擎使用）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | progress_reward + stable_landing_reward | -37.29 | -37.29 | 0.00 | 1000.00 | progress_reward=0.003 stable_landing_reward=4.135 | new_best |
| 2 | progress_reward + stable_landing_reward | -43.99 | -37.29 | -6.70 | 101.20 | progress_reward=0.014 stable_landing_reward=0.138 | no_meaningful_improvement |
| 3 | progress_reward + stable_landing_reward | -82.73 | -37.29 | -45.44 | 858.20 | progress_reward=0.003 stable_landing_reward=4.262 | no_meaningful_improvement |
| 4 | progress_reward + stable_landing_reward | -47.11 | -37.29 | -9.82 | 1000.00 | progress_reward=0.002 stable_landing_reward=5.711 | same_skeleton_persistent_negative_fresh_restart |
| 5 | angle_penalty + potential_diff | -24.46 | -24.46 | 0.00 | 396.35 | angle_penalty=-0.013 potential_diff=0.012 | new_best |
| 6 | angle_penalty + potential_diff + success_bonus | -112.79 | -24.46 | -88.34 | 681.70 | angle_penalty=-0.018 potential_diff=0.008 success_bonus=41.547 | no_meaningful_improvement |
| 7 | angle_penalty + landing_quality_reward + potential_diff | 0.62 | 0.62 | 0.00 | 117.70 | angle_penalty=-0.015 landing_quality_reward=0.006 potential_diff=0.013 | new_best |
| 8 | angle_penalty + landing_reward + potential_diff | -150.57 | 0.62 | -151.19 | 1000.00 | angle_penalty=-0.008 landing_reward=0.846 potential_diff=0.001 | no_meaningful_improvement |
