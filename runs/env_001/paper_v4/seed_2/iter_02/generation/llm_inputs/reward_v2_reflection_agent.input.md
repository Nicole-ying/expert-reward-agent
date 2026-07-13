# Search objective
- target_score: 200.000000
- current_score: -17.897087
- gap_to_target: 217.897087
- target_achievement_ratio: -8.949%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -17.897087）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant states from the next observation (post-transition)
    x_next = next_obs[0]
    y_next = next_obs[1]
    vx_next = next_obs[2]
    vy_next = next_obs[3]
    angle_next = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- Component A: Position Proximity (Main Learning Signal) ----------
    # Dense, bounded reward encouraging the craft to reach (0,0).
    # Uses soft inverse distance so that reward saturates near the target.
    dist = (x_next**2 + y_next**2) ** 0.5
    pos_reward = 1.0 / (1.0 + dist)

    # ---------- Component B: Soft Landing Velocity (Bounded Penalty) ----------
    # Penalise high horizontal/vertical velocity only when close to the ground.
    # Activation gate: high when |y| is small (near platform level), negligible in high sky.
    activation = 1.0 / (1.0 + 10.0 * abs(y_next))
    vel_penalty = -0.1 * activation * (vx_next**2 + vy_next**2)

    # ---------- Component C: Stable Orientation (Quadratic Penalty) ----------
    # Light penalty on body tilt to encourage horizontal attitude.
    angle_penalty = -0.5 * (angle_next ** 2)

    # ---------- Component D: Contact Completion (Joint Condition Proxy) ----------
    # Soft proxy for a “successful two‑leg landing” when multiple conditions are met.
    # Each factor is a continuous [0,1] measure of a desired condition.
    k_x = 5.0
    factor_x = 1.0 / (1.0 + k_x * x_next**2)
    k_y = 5.0
    factor_y = 1.0 / (1.0 + k_y * y_next**2)
    k_vx = 2.0
    factor_vx = 1.0 / (1.0 + k_vx * vx_next**2)
    k_vy = 2.0
    factor_vy = 1.0 / (1.0 + k_vy * vy_next**2)
    k_angle = 3.0
    factor_angle = 1.0 / (1.0 + k_angle * angle_next**2)
    # Contact signals are binary; the product is non‑zero only when both legs touch.
    factor_contact = left_contact * right_contact

    contact_proxy = factor_x * factor_y * factor_vx * factor_vy * factor_angle * factor_contact
    contact_reward = 1.0 * contact_proxy   # up to 1.0 when all conditions are nearly perfect

    # ---------- Total Reward ----------
    total_reward = pos_reward + vel_penalty + angle_penalty + contact_reward

    components = {
        'position_proximity': pos_reward,
        'soft_landing_velocity': vel_penalty,
        'stable_orientation': angle_penalty,
        'contact_completion': contact_reward,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-17.897087, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-45.375937, 21.479454]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| position_proximity | 749.311080 | 99.7% | 99.7% | 100.0% |
| stable_orientation | -1.720382 | -0.2% | 0.2% | 100.0% |
| soft_landing_velocity | -0.332431 | -0.0% | 0.0% | 100.0% |
| contact_completion | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
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

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_completion + position_proximity + soft_landing_velocity + stable_orientation | -17.90 | -17.90 | 0.00 | 1000.00 | contact_completion=0.412 position_proximity=0.746 soft_landing_velocity=-0.002 stable_orientation=-0.016 | new_best |
