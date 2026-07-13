# Search objective
- target_score: 200.000000
- current_score: -388.671871
- gap_to_target: 588.671871
- target_achievement_ratio: -194.336%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -388.671871）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Previous state
    prev_x = obs[0]
    prev_y = obs[1]
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Next state
    x = next_obs[0]
    y = next_obs[1]
    xv = next_obs[2]
    yv = next_obs[3]
    ang = next_obs[4]
    lc = next_obs[6]
    rc = next_obs[7]

    # 1. Approach shaping: reward reduction in distance to pad center
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    approach_shaping = 2.0 * (prev_dist - curr_dist)

    # 2. Descent safety: penalize dangerous high-speed descent near ground
    descent_safety = 0.0
    if yv < 0.0 and y > 0.02:
        total_speed = (xv ** 2 + yv ** 2) ** 0.5
        height_urgency = 1.0 / (1.0 + y)
        descent_safety = -2.0 * total_speed * height_urgency

    # 3. Touchdown bonus: first moment both legs achieve ground contact
    prev_contacts = prev_lc + prev_rc
    next_contacts = lc + rc
    touchdown_bonus = 0.0
    if next_contacts >= 2.0 and prev_contacts < 2.0:
        speed_quality = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_quality = max(0.0, 1.0 - abs(ang) / 0.5)
        touchdown_bonus = 3.0 * (1.0 + speed_quality * angle_quality)

    # 4. Stable landed: ongoing reward for maintaining a successful landing state
    stable_landed = 0.0
    if lc > 0.5 and rc > 0.5 and y < 0.3:
        speed_score = max(0.0, 1.0 - (abs(xv) + abs(yv)))
        angle_score = max(0.0, 1.0 - abs(ang))
        stable_landed = speed_score * angle_score

    # 5. Fuel penalty: small cost for any engine use
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = approach_shaping + descent_safety + touchdown_bonus + stable_landed + fuel_penalty

    components = {
        "approach_shaping": approach_shaping,
        "descent_safety": descent_safety,
        "touchdown_bonus": touchdown_bonus,
        "stable_landed": stable_landed,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-388.671871, len=109.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-499.654146, -304.708700]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| fuel_penalty | -3.597500 | -48.3% | 48.3% | 65.7% |
| descent_safety | -2.805849 | -37.7% | 37.7% | 6.2% |
| approach_shaping | -0.933721 | -12.5% | 14.0% | 100.0% |
| stable_landed | 0.000000 | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主要目标：控制一个二维刚体从上方初始位置尽快到达并稳定停靠在视野中央的目标垫上。  
次要目标：在满足成功停靠的前提下，尽可能少地使用发动机推力（节约燃料/能量）。  
不允许混淆的目标：这是一个典型的到达＋软着陆任务，核心不是维持平衡或长时间存活，而是以安全、低能耗的方式精确到达指定位置。快速性隐含在“as fast as possible”中，但需要与能耗目标协调。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，前6维连续，后2维取0/1但通常保持浮点）
- obs[0]: x_position，相对于目标垫中心的水平距离，reward_usable: true
- obs[1]: y_position，相对于垫面的垂直距离（高度），reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，刚体俯仰角（与垂直/水平方向的偏差），reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿触地标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿触地标志（0/1），reward_usable: true

所有8维均可直接用于奖励计算（位移、速度、姿态、接触状态）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不施加任何推力，惯性飞行
- action 1: left_orientation_engine — 点燃一侧方向引擎（可能产生侧向力矩和/或力，用于调整姿态）
- action 2: main_engine — 点燃主引擎（产生向上的推力，同时可能引起力矩变化）
- action 3: right_orientation_engine — 点燃另一侧方向引擎（与左引擎相反方向）

注意：动作空间是符号化的，但每个动作对“燃料消耗”有隐含影响（0号动作不消耗燃料）。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（“settled”指身体静止并稳定地站立在垫上）是可能的成功终止；但仅凭该条件无法绝对区分成败，因为也可能包含非成功静止（如横躺在地面但物理不活动）。
- failure-like termination: `crash_or_body_contact`（主体部分与地面或其它障碍激烈碰撞）、`horizontal_position_outside_viewport`（偏离目标太远）。
- ambiguous termination: 所有的终止条件均不显式标记成功/失败，需结合观测（位置、速度、姿态、双腿触地）判断。
- truncation: 源未提供步数截断信息，假设环境无显式步数限制，但从实际训练安全考虑通常会有时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空字典，无可用字段）
- forbidden_or_uncertain_info_fields: 没有任何 info 字段被允许供奖励使用，因此不能依赖任何隐藏在 info 中的成功标志。

## 7. 可用于奖励函数的信号
- position: 相对垫子的水平距离 x_position (obs[0])，高度 y_position (obs[1])
- velocity: 水平速度 x_velocity (obs[2])，垂直速度 y_velocity (obs[3])
- orientation: 身体角度 body_angle (obs[4])，角速度 angular_velocity (obs[5])
- contact: 左腿触碰 left_support_contact (obs[6])，右腿触碰 right_support_contact (obs[7])
- action/engine: 动作类型 (action)，可以推断燃料消耗（非零动作消耗）
- other: 从 next_obs 可计算速度变化、当前信息；终止可与接触模式结合判断着陆是否完全稳定。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_penalty + landing_reward + proximity | -70.35 | -70.35 | 0.00 | 1000.00 | fuel_penalty=-0.041 landing_reward=0.295 proximity=1.568 | new_best |
| 2 | fuel_penalty + landing_reward + progress | -129.32 | -70.35 | -58.97 | 69.65 | fuel_penalty=-0.003 landing_reward=0.002 progress=0.080 | no_meaningful_improvement |
| 3 | fuel_penalty + landing_reward + proximity | -26.30 | -26.30 | 0.00 | 951.00 | fuel_penalty=-0.034 landing_reward=2.216 proximity=0.072 | new_best |
| 4 | fuel_penalty + landing_reward + proximity | -120.27 | -26.30 | -93.97 | 68.35 | fuel_penalty=-0.006 landing_reward=0.087 proximity=0.053 | no_meaningful_improvement |
| 5 | approach_improvement + contact_event + fuel_penalty + progress_delta | -123.86 | -26.30 | -97.57 | 68.30 | approach_improvement=0.028 contact_event=0.022 fuel_penalty=-0.006 progress_delta=0.016 | no_meaningful_improvement |
| 6 | descent_safety + fuel_penalty + safe_proximity + stable_landed + touchdown_bonus | -18.28 | -18.28 | 0.00 | 911.60 | descent_safety=-0.114 fuel_penalty=-0.040 safe_proximity=0.502 stable_landed=0.003 touchdown_bonus=0.004 | new_best |
| 7 | approach_shaping + descent_safety + fuel_penalty + stable_landed + touchdown_bonus | -388.67 | -18.28 | -370.39 | 109.50 | approach_shaping=-0.016 descent_safety=-0.078 fuel_penalty=-0.040 stable_landed=0.001 touchdown_bonus=0.001 | no_meaningful_improvement |
