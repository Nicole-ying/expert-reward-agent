# Duplicate reward retry
The previous generation duplicated iter 5 (runs\env_001\paper_v4\seed_0\iter_05\generation\reward_v5.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract previous state
    prev_x = obs[0]
    prev_y = obs[1]
    prev_xv = obs[2]
    prev_yv = obs[3]
    prev_ang = obs[4]
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Extract next state
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_xv = next_obs[2]
    next_yv = next_obs[3]
    next_ang = next_obs[4]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    # 1. Progress delta: how much closer to the target center
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress_delta = prev_dist - next_dist  # positive = approaching, negative = receding

    # 2. Approach improvement: change in landing-quality potential
    def landing_quality(y, lc, rc, xv, yv, ang, threshold=0.5):
        if y < threshold:
            contact = lc + rc
            total_speed = abs(xv) + abs(yv)
            speed_ok = max(0.0, 1.0 - total_speed / 2.0)
            angle_ok = max(0.0, 1.0 - abs(ang) / 0.5)
            return contact * speed_ok * angle_ok
        return 0.0

    prev_quality = landing_quality(prev_y, prev_lc, prev_rc, prev_xv, prev_yv, prev_ang)
    next_quality = landing_quality(next_y, next_lc, next_rc, next_xv, next_yv, next_ang)
    approach_improvement = 2.0 * (next_quality - prev_quality)

    # 3. Contact event: reward first moment both legs touch ground
    prev_contacts = prev_lc + prev_rc
    next_contacts = next_lc + next_rc
    contact_event = 2.0 if (next_contacts == 2 and prev_contacts < 2) else 0.0

    # 4. Fuel efficiency
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = progress_delta + approach_improvement + contact_event + fuel_penalty

    components = {
        "progress_delta": progress_delta,
        "approach_improvement": approach_improvement,
        "contact_event": contact_event,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```

# Search objective
- target_score: 200.000000
- current_score: -123.863893
- gap_to_target: 323.863893
- target_achievement_ratio: -61.932%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -123.863893）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract previous state
    prev_x = obs[0]
    prev_y = obs[1]
    prev_xv = obs[2]
    prev_yv = obs[3]
    prev_ang = obs[4]
    prev_lc = obs[6]
    prev_rc = obs[7]

    # Extract next state
    next_x = next_obs[0]
    next_y = next_obs[1]
    next_xv = next_obs[2]
    next_yv = next_obs[3]
    next_ang = next_obs[4]
    next_lc = next_obs[6]
    next_rc = next_obs[7]

    # 1. Progress delta: how much closer to the target center
    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5
    progress_delta = prev_dist - next_dist  # positive = approaching, negative = receding

    # 2. Approach improvement: change in landing-quality potential
    def landing_quality(y, lc, rc, xv, yv, ang, threshold=0.5):
        if y < threshold:
            contact = lc + rc
            total_speed = abs(xv) + abs(yv)
            speed_ok = max(0.0, 1.0 - total_speed / 2.0)
            angle_ok = max(0.0, 1.0 - abs(ang) / 0.5)
            return contact * speed_ok * angle_ok
        return 0.0

    prev_quality = landing_quality(prev_y, prev_lc, prev_rc, prev_xv, prev_yv, prev_ang)
    next_quality = landing_quality(next_y, next_lc, next_rc, next_xv, next_yv, next_ang)
    approach_improvement = 2.0 * (next_quality - prev_quality)

    # 3. Contact event: reward first moment both legs touch ground
    prev_contacts = prev_lc + prev_rc
    next_contacts = next_lc + next_rc
    contact_event = 2.0 if (next_contacts == 2 and prev_contacts < 2) else 0.0

    # 4. Fuel efficiency
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = progress_delta + approach_improvement + contact_event + fuel_penalty

    components = {
        "progress_delta": progress_delta,
        "approach_improvement": approach_improvement,
        "contact_event": contact_event,
        "fuel_penalty": fuel_penalty
    }

    return (float(total_reward), components)
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-123.863893, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335003, -103.388556]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| approach_improvement | 1.759462 | 39.1% | 39.1% | 1.6% |
| contact_event | 1.400000 | 31.1% | 31.1% | 1.0% |
| progress_delta | 1.114678 | 24.8% | 25.6% | 100.0% |
| fuel_penalty | -0.185000 | -4.1% | 4.1% | 5.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
