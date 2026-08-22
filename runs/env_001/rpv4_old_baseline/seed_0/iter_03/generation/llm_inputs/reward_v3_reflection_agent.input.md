# 1. Search objective
- target_score: 200.000000
- current_score: -98.901539
- gap_to_target: 298.901539

# 2. Current reward program (score: -98.901539)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========================== 权重参数 ==========================
    w_progress = 2.0        # 距离缩减的正面奖励
    w_vx = 0.001            # 水平速度惩罚（小权重，避免过度压制接近行为）
    w_vy = 0.001            # 垂直速度惩罚
    w_angle = 0.005         # 姿态角惩罚
    w_ang_vel = 0.001       # 角速度惩罚
    w_action = 0.001        # 引擎使用惩罚（极小，允许必要机动）
    w_landing = 150.0       # 精确软着陆奖励（每步，条件严格）

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

    # ========================== 组件 B：稳定性/安全约束（弱惩罚） ==========================
    penalty_vx = -w_vx * (vx_next ** 2)
    penalty_vy = -w_vy * (vy_next ** 2)
    penalty_angle = -w_angle * (angle_next ** 2)
    penalty_ang_vel = -w_ang_vel * (ang_vel_next ** 2)
    stability_penalty = penalty_vx + penalty_vy + penalty_angle + penalty_ang_vel

    # ========================== 组件 C：引擎使用惩罚 ==========================
    if action != 0:
        action_penalty = -w_action
    else:
        action_penalty = 0.0

    # ========================== 组件 D：软着陆大额奖励 ==========================
    # 条件：至少一个支撑脚接触，且位置、速度均接近零，角度/角速度也可考虑但不强制
    any_contact = (left_contact > 0.5 or right_contact > 0.5)
    near_target = (abs(x_next) < 0.1 and abs(y_next) < 0.1)
    low_speed   = (abs(vx_next) < 0.2 and abs(vy_next) < 0.5)
    if any_contact and near_target and low_speed:
        landing_reward = w_landing
    else:
        landing_reward = 0.0

    # ========================== 总奖励 ==========================
    total_reward = progress_reward + stability_penalty + action_penalty + landing_reward

    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'action_penalty': action_penalty,
        'landing_reward': landing_reward
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-98.901539, len=876.950000, terminated=3/20, truncated=17/20, reward_errors=0
score_range=[-170.502936, 290.659645]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 450.000000 | 99.4% | 99.4% | 0.3% |
| progress_reward | 0.452978 | 0.1% | 0.4% | 100.0% |
| action_penalty | -0.874150 | -0.2% | 0.2% | 99.7% |
| stability_penalty | -0.090665 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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