# 1. Search objective
- target_score: 200.000000
- current_score: -122.830710
- gap_to_target: 322.830710

# 2. Current reward program (score: -122.830710)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    dist = (x * x + y * y) ** 0.5 + 1e-8

    # 1. Radial velocity reward: positive when moving toward target
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y
    radial_reward = 3.0 * radial_vel

    # 2. Velocity shaping: penalise horizontal drift and encourage controlled descent
    if y > 0.3:
        desired_vy = -0.4
    elif y > 0.1:
        desired_vy = -0.25
    else:
        desired_vy = -0.05
    vy_error = (vy - desired_vy) ** 2
    vx_error = vx ** 2
    vel_penalty = -0.8 * vx_error - 0.6 * vy_error

    # 3. Attitude penalty: stay upright
    angle_penalty = -2.0 * abs(angle)

    # 4. Engine usage penalty
    if action == 0:
        engine_penalty = 0.0
    elif action in (1, 3):   # orientation engines
        engine_penalty = -0.2
    elif action == 2:        # main engine
        engine_penalty = -0.5
    else:
        engine_penalty = 0.0

    # 5. Small proximity bonus, only when descending
    if y < 0.5 and dist < 1.5 and vy < -0.1:
        proximity_bonus = 0.5 * (1.5 - dist) * max(0.0, 0.5 - y)
    else:
        proximity_bonus = 0.0

    # 6. Descent progress: reward altitude reduction
    descent_reward = 2.0 * max(0.0, obs[1] - next_obs[1])

    # 7. Landing and contact bonuses / penalties (modified: no extreme crash penalty,
    #    continuous speed/angle penalty, high success bonuses preserved)
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x) < 0.4 and y < 0.3)

    if any_contact:
        speed = (vx * vx + vy * vy) ** 0.5
        # Smooth penalty for high speed and large angle on contact
        contact_penalty = -10.0 * speed - 10.0 * abs(angle)
        if full_contact and near_target:
            if speed < 0.6 and abs(angle) < 0.5:
                landing_bonus = 500.0 + contact_penalty   # perfect landing
            else:
                landing_bonus = 200.0 + contact_penalty   # acceptable landing near target
        else:
            landing_bonus = contact_penalty
    else:
        landing_bonus = 0.0

    # 8. Per-step time penalty
    time_penalty = -0.05

    total_reward = (radial_reward + vel_penalty + angle_penalty +
                    engine_penalty + proximity_bonus + descent_reward +
                    landing_bonus + time_penalty)

    components = {
        'radial_reward': radial_reward,
        'vel_penalty': vel_penalty,
        'angle_penalty': angle_penalty,
        'engine_penalty': engine_penalty,
        'proximity_bonus': proximity_bonus,
        'descent_reward': descent_reward,
        'landing_bonus': landing_bonus,
        'time_penalty': time_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-122.830710, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-147.719906, -99.206220]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 201.363514 | 45.1% | 54.9% | 3.0% |
| radial_reward | 133.090750 | 29.8% | 31.8% | 100.0% |
| vel_penalty | -38.642529 | -8.6% | 8.6% | 100.0% |
| angle_penalty | -11.693155 | -2.6% | 2.6% | 100.0% |
| time_penalty | -3.415000 | -0.8% | 0.8% | 100.0% |
| descent_reward | 2.864226 | 0.6% | 0.6% | 94.3% |
| proximity_bonus | 2.135681 | 0.5% | 0.5% | 20.7% |
| engine_penalty | -0.580000 | -0.1% | 0.1% | 4.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
这是一个 2D 飞行器/着陆器任务。主体从画面顶部中央附近出发，受到随机初始力作用。**主要目标**是尽快飞到并稳定在中央的目标着陆垫上，同时尽可能少用引擎推力。智能体必须学会：平滑接近目标、减小速度、保持水平姿态，并以安全接触方式着陆。**次要目标**是节约燃料（即少用引擎）。不应将目标分解为纯导航或纯生存，到达并稳定着陆是本环境的唯一核心目标，燃料效率是附加在相同轨迹上的性能优化。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32（推测）
- obs[0]: `x_position_relative_to_target`，水平坐标（相对目标垫），reward_usable: true
- obs[1]: `y_position_relative_to_pad_height`，垂直坐标（相对垫高度），reward_usable: true
- obs[2]: `x_velocity`，水平线速度，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，reward_usable: true
- obs[4]: `body_angle`，机体方向角（弧度？），reward_usable: true
- obs[5]: `angular_velocity`，角速度，reward_usable: true
- obs[6]: `left_support_contact`，左侧支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: `right_support_contact`，右侧支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作 0: `no_engine` —— 所有引擎关闭，不施加推力。
- 动作 1: `left_orientation_engine` —— 点燃左姿态引擎，产生改变姿态的力。
- 动作 2: `main_engine` —— 点燃主引擎（可能产生向上的主推力）。
- 动作 3: `right_orientation_engine` —— 点燃右姿态引擎，产生反向改变姿态的力。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: 无明确的成功终止类型，但 `body_not_awake_or_settled` 可能对应于成功着陆并稳定在目标垫上，亦可能在到达垫之前就休眠造成失败，需结合位置判断。
- **failure-like termination**: `crash_or_body_contact`（硬碰撞或非目标垫的接触）和 `horizontal_position_outside_viewport`（水平超出画布）显然是失败终止。
- **ambiguous termination**: `body_not_awake_or_settled` 本身不能直接区分成功或失败。
- **truncation**: 未提供任何截断条件。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空）
- explicit_failure_flag_available: false （info 为空）
- allowed_info_fields: 无（info 是空字典 `{}`）
- forbidden_or_uncertain_info_fields: 无（所有 info 字段均不可用）

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（x）, `obs[1]`（y），以及对应的 next_obs 值，可构造距离、接近速度等。
- **速度**：`obs[2]`（vx）, `obs[3]`（vy），可用于惩罚过快着陆或水平漂移。
- **姿态与角速度**：`obs[4]`（角度）, `obs[5]`（角速度），可用于鼓励水平姿态和稳定性。
- **接触标志**：`obs[6]`, `obs[7]`，可判断是否与垫接触（但无法区分是目标垫还是其他表面，只能结合位置估计成功着陆）。
- **动作/引擎使用**：`action` 本身，0 为无推力，1/2/3 表示使用了引擎，可用于惩罚或奖励节油。
- **其他**：无。