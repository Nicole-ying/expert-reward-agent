# 1. Search objective
- target_score: 200.000000
- current_score: -352.523770
- gap_to_target: 552.523770

# 2. Current reward program (score: -352.523770)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 权重参数 ==========================
    w_progress = 5.0             # 距离缩减奖励，加大引导力
    w_time = -0.005              # 极小时间惩罚
    w_action_orient = -0.1       # 姿态引擎惩罚
    w_action_main = -0.2         # 主引擎惩罚

    # 接近目标区稳定性约束权重，仅当 dist < 2.0 时生效
    w_vx_near = 0.01
    w_vy_near = 0.01
    w_angle_near = 0.05
    w_ang_vel_near = 0.01

    w_landing = 200.0            # 成功着陆奖励
    w_crash = -100.0             # 非目标区接触惩罚

    # ========================== 观察量解析 ==========================
    x_cur  = obs[0]
    y_cur  = obs[1]
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========================== 组件 A：距离缩减奖励 ==========================
    dist_before = (x_cur**2 + y_cur**2) ** 0.5
    dist_after  = (x_next**2 + y_next**2) ** 0.5
    progress_reward = w_progress * (dist_before - dist_after)

    # ========================== 组件 B：时间惩罚 ==========================
    time_penalty = w_time  # 常数，每步相同

    # ========================== 组件 C：接近目标区稳定性约束 ==========================
    # 仅当处于目标附近（dist_after < 2.0）时，才对速度、角度施加轻微惩罚，
    # 引导减速摆正；远距离不做限制，允许自由机动。
    near_region = max(0.0, 1.0 - dist_after / 2.0)  # dist<2时线性增大到1
    penalty_vx_near = -w_vx_near * near_region * abs(vx_next)
    penalty_vy_near = -w_vy_near * near_region * abs(vy_next)
    penalty_angle_near = -w_angle_near * near_region * abs(angle_next)
    penalty_ang_vel_near = -w_ang_vel_near * near_region * abs(ang_vel_next)
    stability_penalty = penalty_vx_near + penalty_vy_near + penalty_angle_near + penalty_ang_vel_near

    # ========================== 组件 D：引擎使用惩罚 ==========================
    if action == 0:
        action_penalty = 0.0
    elif action in (1, 3):          # 姿态引擎
        action_penalty = w_action_orient
    elif action == 2:               # 主引擎
        action_penalty = w_action_main
    else:
        action_penalty = 0.0

    # ========================== 组件 E：成功着陆奖励 ==========================
    # 判断两脚是否充分接触（采用逻辑与保证稳定着陆）
    full_contact = (left_contact > 0.5 and right_contact > 0.5)
    near_target = (abs(x_next) < 0.2 and abs(y_next) < 0.2)
    low_speed   = (abs(vx_next) < 0.5 and abs(vy_next) < 0.5)
    upright = abs(angle_next) < 0.3
    if full_contact and near_target and low_speed and upright:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # ========================== 组件 F：非目标区接触惩罚 ==========================
    # 任何单脚或双脚接触，但不在目标区域即视为危险接触
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    if any_contact and not near_target:
        crash_penalty = w_crash
    else:
        crash_penalty = 0.0

    # ========================== 总奖励 ==========================
    total_reward = progress_reward + time_penalty + stability_penalty + action_penalty + landing_reward + crash_penalty

    components = {
        'progress_reward': progress_reward,
        'time_penalty': time_penalty,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'landing_reward': landing_reward,
        'crash_penalty': crash_penalty
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-352.523770, len=66.950000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-397.932323, -310.312542]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| crash_penalty | -35.000000 | -75.5% | 75.5% | 0.5% |
| progress_reward | 5.615138 | 12.1% | 12.4% | 100.0% |
| stability_penalty | -2.844977 | -6.1% | 6.1% | 100.0% |
| action_penalty | -2.420000 | -5.2% | 5.2% | 36.1% |
| time_penalty | -0.334750 | -0.7% | 0.7% | 100.0% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

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