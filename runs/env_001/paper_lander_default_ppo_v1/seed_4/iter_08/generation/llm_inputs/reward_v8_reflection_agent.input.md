# Search objective
- target_score: 200.000000
- current_score: -24.032875
- gap_to_target: 224.032875
- target_achievement_ratio: -12.016%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -24.032875）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, left_contact, right_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nleft_contact, nright_contact = next_obs

    # ----------------------------------------------------------------
    # 1. Goal proximity: potential-based delta shaping
    #    Positive when moving toward target, negative when moving away
    # ----------------------------------------------------------------
    distance = (x**2 + y**2)**0.5
    next_distance = (nx**2 + ny**2)**0.5
    proximity_reward = 2.0 * (distance - next_distance)

    # ----------------------------------------------------------------
    # 2. Orientation penalty: keep body upright and stable
    # ----------------------------------------------------------------
    orientation_penalty = -0.1 * (angle**2) - 0.1 * (angvel**2)

    # ----------------------------------------------------------------
    # 3. Soft landing: punish high velocities when near the pad
    # ----------------------------------------------------------------
    proximity_gate = 1.0 / (1.0 + 5.0 * distance)
    speed_sq = vx**2 + vy**2
    speed_penalty_gated = -0.5 * speed_sq * proximity_gate

    # ----------------------------------------------------------------
    # 4. Settlement improvement: proper potential-based delta
    #    Uses full delta (NOT max(0, delta)) to prevent exploitation
    #    by oscillation. Total per episode is strictly bounded:
    #    150.0 * (final_settled - initial_settled) <= 150.0
    #    Positive when becoming more settled, negative when regressing,
    #    forcing the agent to truly complete the landing for net gain.
    # ----------------------------------------------------------------
    # Current settled score (from obs)
    current_avg_contact = (left_contact + right_contact) / 2.0
    current_prox_gate = 1.0 / (1.0 + 5.0 * distance)
    current_speed_sq = vx**2 + vy**2
    current_stillness = 1.0 / (1.0 + 10.0 * (current_speed_sq + angvel**2))
    current_settled = current_avg_contact * current_prox_gate * current_stillness

    # Next settled score (from next_obs)
    next_avg_contact = (nleft_contact + nright_contact) / 2.0
    next_prox_gate = 1.0 / (1.0 + 5.0 * next_distance)
    next_speed_sq = nvx**2 + nvy**2
    next_stillness = 1.0 / (1.0 + 10.0 * (next_speed_sq + nangvel**2))
    next_settled = next_avg_contact * next_prox_gate * next_stillness

    # Full potential-based delta: symmetric, no exploitation by oscillation
    settlement_bonus = 150.0 * (next_settled - current_settled)

    # ----------------------------------------------------------------
    # Combine
    # ----------------------------------------------------------------
    total_reward = (
        proximity_reward +
        orientation_penalty +
        speed_penalty_gated +
        settlement_bonus
    )

    components = {
        "proximity_reward": proximity_reward,
        "orientation_penalty": orientation_penalty,
        "speed_penalty_gated": speed_penalty_gated,
        "settlement_bonus": settlement_bonus
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-24.032875, len=148.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-118.838826, 261.583534]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settlement_bonus | 25.240861 | 41.7% | 80.9% | 37.4% |
| speed_penalty_gated | -8.926664 | -14.8% | 14.8% | 99.6% |
| proximity_reward | 2.081760 | 3.4% | 3.7% | 99.6% |
| orientation_penalty | -0.353664 | -0.6% | 0.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是一个 2D 飞行器 / 着陆器轨迹优化问题。  
主体从视野顶部中央附近出发，并带有随机初速度。核心目标是**尽快到达并稳定停靠在画面中央的目标平台上、且保持姿态平稳**；次要目标是在此过程中尽量减少引擎推力消耗。需要关键控制能力：接近目标、减速、维持直立姿态并通过左右支撑点与平台安全接触。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: `float32`  

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|----------------|
| 0 | x_position | 相对于目标平台的水平坐标 | true |
| 1 | y_position | 相对于平台高度的高度坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体倾角（方向角） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑接触标志（0/1） | true |
| 7 | right_support_contact | 右支撑接触标志（0/1） | true |

注：接触标志虽然在环境中可能是 `1.0` 或 `0.0`，但该维度对奖励设计有用。

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  

| 动作编号 | 名称 | 含义 | 说明 |
|----------|------|------|------|
| 0 | no_engine | 不点火 | 被动滑行 |
| 1 | left_orientation_engine | 左侧姿控引擎 | 产生角度变化 / 姿态调整 |
| 2 | main_engine | 主引擎 | 产生主要向下推力，影响垂直速度 |
| 3 | right_orientation_engine | 右侧姿控引擎 | 产生相反侧姿态调整 |

本离散动作空间没有幅度控制，每次执行一个动作施加固定强度的冲量。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 当主体运动停止且可能稳定在平台上时触发。该条件**并非明确只包含成功**，可能包含因其他原因静止（如坠毁后静止），需要结合位置与接触状态判断。
- **failure-like termination**:  
  `crash_or_body_contact` —— 主体与非目标地面/结构物接触导致坠毁；  
  `horizontal_position_outside_viewport` —— 主体水平飘出视野边界。
- **ambiguous termination**:  
  `body_not_awake_or_settled` 本身未直接区分成功与失败。
- **truncation**:  
  来源于 `step` 源码中的 `terminated` 标志，`truncated` 始终为 `False`，没有超时截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空（`step` 返回的 `info` 为空字典 `{}`）  
- **forbidden_or_uncertain_info_fields**: 所有 `info` 内未声明的字段，包括假设的 `success`、`failure`、`termination_reason` 等均不可用。

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（相对平台坐标，可用以引导接近与保持在目标上方）
- **velocity**: `x_velocity`, `y_velocity`（减速靠近平台）
- **orientation**: `body_angle`, `angular_velocity`（维持姿态平稳）
- **contact**: `left_support_contact`, `right_support_contact`（稳定接触检测）
- **action/engine**: 可以针对推力动作施加惩罚（如 `action != 0`）以鼓励节省燃料
- **other**: 无额外信号。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | contact_bonus + orientation_penalty + proximity_reward + speed_penalty_gated | -65.97 | -65.97 | 0.00 | 72.00 | contact_bonus=0.001 orientation_penalty=-0.006 proximity_reward=-0.971 speed_penalty_gated=-0.166 | new_best |
| 2 | contact_bonus + orientation_penalty + proximity_reward + speed_penalty_gated | 126.73 | 126.73 | 0.00 | 749.25 | contact_bonus=0.276 orientation_penalty=-0.004 proximity_reward=0.004 speed_penalty_gated=-0.012 | new_best |
| 3 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 187.04 | 187.04 | 0.00 | 715.90 | orientation_penalty=-0.004 proximity_reward=0.005 settlement_bonus=0.934 speed_penalty_gated=-0.016 | new_best |
| 4 | attitude_penalty + direction_reward + event_bonus + fuel_penalty + soft_landing_proxy | 168.77 | 187.04 | -18.27 | 517.55 | attitude_penalty=-0.013 direction_reward=0.045 event_bonus=0.808 fuel_penalty=-0.007 soft_landing_proxy=0.373 | no_meaningful_improvement |
| 5 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 197.60 | 197.60 | 0.00 | 669.50 | orientation_penalty=-0.004 proximity_reward=0.005 settlement_bonus=0.942 speed_penalty_gated=-0.016 | new_best |
| 6 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | 89.07 | 197.60 | -108.54 | 805.90 | orientation_penalty=-0.005 proximity_reward=0.007 settlement_bonus=15.403 speed_penalty_gated=-0.023 | no_meaningful_improvement |
| 7 | orientation_penalty + proximity_reward + settlement_bonus + speed_penalty_gated | -24.03 | 197.60 | -221.64 | 148.15 | orientation_penalty=-0.004 proximity_reward=0.029 settlement_bonus=0.426 speed_penalty_gated=-0.155 | no_meaningful_improvement |
