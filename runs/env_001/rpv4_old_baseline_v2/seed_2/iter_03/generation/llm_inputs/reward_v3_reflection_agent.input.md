# 1. Search objective
- target_score: 200.000000
- current_score: -143.494333
- gap_to_target: 343.494333

# 2. Current reward program (score: -143.494333)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous and current state
    x_prev = obs[0]
    y_prev = obs[1]

    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- A: Descent Incentive (main learning signal) ----------
    # Reward each step of downward progress toward the platform.
    descent = y_prev - y
    descent_reward = 3.0 * max(0.0, descent)

    # ---------- B: Altitude Proximity (auxiliary gradient near ground) ----------
    # Small, saturating reward for simply being near platform level.
    alt_reward = 0.3 / (1.0 + 3.0 * abs(y))

    # ---------- C: Horizontal Centering (gated by altitude) ----------
    # Only reward x-alignment when the craft is close to the platform.
    near_ground = 1.0 / (1.0 + 5.0 * abs(y))
    center_reward = 0.3 * near_ground / (1.0 + 2.0 * abs(x))

    # ---------- D: Soft Landing Velocity Penalty (gated) ----------
    # Penalise high speed when close to ground, with emphasis on vertical.
    speed_penalty = -1.0 * near_ground * (0.5 * vx * vx + vy * vy)

    # ---------- E: Attitude Penalty (gated by altitude) ----------
    # Penalise tilt and rotation when close to the platform.
    attitude_penalty = -0.3 * near_ground * (angle * angle + 0.5 * angvel * angvel)

    # ---------- F: Contact Success Bonus (one-time landing reward) ----------
    # Large bonus when both legs touch, scaled by landing quality.
    contact = left_contact * right_contact
    landing_quality = 1.0 / (1.0 + 10.0 * (vx * vx + vy * vy) + 3.0 * angle * angle + 3.0 * abs(x))
    contact_bonus = 25.0 * contact * landing_quality

    # ---------- G: Step Cost (anti-hovering pressure) ----------
    step_cost = -0.03

    # ---------- H: Engine Efficiency ----------
    if action == 0:
        engine_cost = 0.0
    elif action == 2:
        engine_cost = -0.05
    else:
        engine_cost = -0.02

    total_reward = (
        descent_reward
        + alt_reward
        + center_reward
        + speed_penalty
        + attitude_penalty
        + contact_bonus
        + step_cost
        + engine_cost
    )

    components = {
        'descent_reward': descent_reward,
        'altitude_proximity': alt_reward,
        'horizontal_centering': center_reward,
        'speed_penalty': speed_penalty,
        'attitude_penalty': attitude_penalty,
        'contact_bonus': contact_bonus,
        'step_cost': step_cost,
        'engine_cost': engine_cost,
    }

    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-143.494333, len=590.650000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-189.979339, -13.311123]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| altitude_proximity | 61.638530 | 46.1% | 46.1% | 100.0% |
| horizontal_centering | 25.132068 | 18.8% | 18.8% | 100.0% |
| engine_cost | -21.977000 | -16.5% | 16.5% | 99.3% |
| step_cost | -17.719500 | -13.3% | 13.3% | 100.0% |
| speed_penalty | -4.113149 | -3.1% | 3.1% | 100.0% |
| descent_reward | 2.864747 | 2.1% | 2.1% | 86.6% |
| attitude_penalty | -0.139886 | -0.1% | 0.1% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
### Overall score
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 4. Environment facts
## 1. 任务目标
该环境是一个 2D 飞行器着陆任务。飞行器从目标着陆平台上方区域出发，受到随机初始扰动力。核心目标是引导飞行器尽快接近并稳定地停在中央着陆平台上，并最大限度减少引擎使用。成功的着陆应满足：位置靠近平台中心、相对垂直速度很小、身体姿态平稳、且两个支撑部件都与平台接触。附属目标是降低燃料消耗（即减少引擎推力动作的使用）并缩短耗时。任务**不**是简单的平衡或存活，而是以到达指定位置为根本驱动。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: 默认 `float32`（其中 contact 标志为 0.0 / 1.0）
- 每一维含义及奖励可用性：

| 维度索引 | 名称 | 含义 | reward_usable |
|-----------|------|------|---------------|
| 0 | `x_position` | 飞行器相对于着陆平台的水平坐标 | true |
| 1 | `y_position` | 飞行器相对于平台高度的垂直坐标 | true |
| 2 | `x_velocity` | 水平线速度 | true |
| 3 | `y_velocity` | 垂直线速度 | true |
| 4 | `body_angle` | 机体倾斜角 | true |
| 5 | `angular_velocity` | 角速度 | true |
| 6 | `left_support_contact` | 左支撑腿是否接触地面（0/1） | true |
| 7 | `right_support_contact` | 右支撑腿是否接触地面（0/1） | true |

所有维度均可安全用于奖励函数计算，不存在需禁用的字段。

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- 动作含义：

| 动作索引 | 名称 | 含义 |
|----------|------|------|
| 0 | `no_engine` | 无任何推力输出 |
| 1 | `left_orientation_engine` | 开启左侧转向引擎（产生逆时针旋转力矩及少量推力） |
| 2 | `main_engine` | 开启主引擎（产生强大的向下推力，抵消重力） |
| 3 | `right_orientation_engine` | 开启右侧转向引擎（产生顺时针旋转力矩及少量推力） |

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 当飞行器身体不再活跃（进入休眠）且已基本静止时触发。通常由稳定着陆引发。
- **failure-like termination**: `crash_or_body_contact` —— 飞行器与地面/平台发生不当碰撞（如侧面触地、强烈冲击），视为坠毁；`horizontal_position_outside_viewport` —— 水平位置超出视觉窗口，表示偏离过大。
- **ambiguous termination**: 无。所有终止条件均可明确区分为成功或失败。
- **truncation**: 环境描述未提及最大步数截断，但实际部署时可能存在；truncation 不能被直接当作成功/失败信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （`info` 字典为空，无直接标志位）
- explicit_failure_flag_available: **false** （同上）
- allowed_info_fields: **无** 额外允许字段（`info = {}`）
- forbidden_or_uncertain_info_fields: 所有未在 step source 中出现的 `info` 键均不可用，包括但不限于 `success`、`failure`、`termination_reason`。

因此，奖励函数**必须**仅依赖 `obs`、`action`、`next_obs` 以及终止状态来隐式推断任务成果，不能依赖外部成功/失败标记。

## 7. 可用于奖励函数的信号
- **位置信号**: `x_position`, `y_position` （相对目标，可直接计算距离）
- **速度信号**: `x_velocity`, `y_velocity` （衡量平缓着陆的关键）
- **姿态信号**: `body_angle` （水平姿态维稳）
- **角速度**: `angular_velocity`
- **接触信号**: `left_support_contact`, `right_support_contact` （软着陆的最后确认）
- **动作/引擎信号**: 动作索引可识别是否使用主引擎或转向引擎，用于燃料惩罚。
- **其他潜在信号**: 可从 `obs` 和 `next_obs` 差分得到加速度等，但需谨慎引入。