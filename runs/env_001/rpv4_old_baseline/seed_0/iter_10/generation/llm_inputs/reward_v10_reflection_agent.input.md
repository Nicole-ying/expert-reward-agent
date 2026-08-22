# 1. Search objective
- target_score: 200.000000
- current_score: -118.059196
- gap_to_target: 318.059196

# 2. Current reward program (score: -118.059196)
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

    # 1. 径向速度奖励：始终鼓励向目标移动
    dir_x = -x / dist
    dir_y = -y / dist
    radial_vel = vx * dir_x + vy * dir_y
    radial_reward = 2.0 * radial_vel

    # 2. 水平漂移惩罚：抑制侧向速度
    vx_penalty = -0.3 * abs(vx)

    # 3. 垂直速度引导：根据高度把vy维持在一个安全区间
    if y > 0.5:
        vy_low, vy_high = -0.5, -0.1
    elif y > 0.1:
        vy_low, vy_high = -0.3, -0.05
    else:
        vy_low, vy_high = -0.15, -0.02

    if vy < vy_low:
        vy_penalty = -0.5 * (vy_low - vy)   # 下降太快
    elif vy > vy_high:
        vy_penalty = -0.5 * (vy - vy_high)  # 上升或太慢
    else:
        vy_penalty = 0.0

    # 4. 姿态惩罚：轻微惩罚倾斜
    angle_penalty = -0.5 * abs(angle)

    # 5. 引擎使用惩罚：鼓励节油
    if action == 0:
        engine_penalty = 0.0
    elif action in (1, 3):
        engine_penalty = -0.15
    elif action == 2:
        engine_penalty = -0.4
    else:
        engine_penalty = 0.0

    # 6. 水平位置惩罚：防止飞出视口
    x_penalty = -0.2 * abs(x)

    # 7. 下降进度奖励：适度鼓励高度降低
    descent_reward = 0.5 * max(0.0, obs[1] - next_obs[1])

    # 8. 着陆奖励（接触时触发）
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x) < 0.4 and y < 0.3)

    if any_contact:
        speed = (vx * vx + vy * vy) ** 0.5
        contact_penalty = -2.0 * speed - 2.0 * abs(angle)
        if full_contact and near_target:
            if speed < 0.3 and abs(angle) < 0.3:
                landing_bonus = 400.0 + contact_penalty
            else:
                landing_bonus = 150.0 + contact_penalty
        else:
            landing_bonus = contact_penalty
    else:
        landing_bonus = 0.0

    # 9. 时间惩罚：鼓励尽快完成任务
    time_penalty = -0.05

    total_reward = (radial_reward + vx_penalty + vy_penalty + angle_penalty +
                    engine_penalty + x_penalty + descent_reward +
                    landing_bonus + time_penalty)

    components = {
        'radial_reward': radial_reward,
        'vx_penalty': vx_penalty,
        'vy_penalty': vy_penalty,
        'angle_penalty': angle_penalty,
        'engine_penalty': engine_penalty,
        'x_penalty': x_penalty,
        'descent_reward': descent_reward,
        'landing_bonus': landing_bonus,
        'time_penalty': time_penalty,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-118.059196, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.398763, -95.050266]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 204.129179 | 59.3% | 61.6% | 3.1% |
| radial_reward | 88.889133 | 25.8% | 27.6% | 100.0% |
| vy_penalty | -19.152378 | -5.6% | 5.6% | 83.8% |
| vx_penalty | -8.537503 | -2.5% | 2.5% | 100.0% |
| time_penalty | -3.420000 | -1.0% | 1.0% | 100.0% |
| angle_penalty | -2.877625 | -0.8% | 0.8% | 100.0% |
| x_penalty | -2.100649 | -0.6% | 0.6% | 100.0% |
| descent_reward | 0.716764 | 0.2% | 0.2% | 94.3% |
| engine_penalty | -0.577500 | -0.2% | 0.2% | 5.6% |

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