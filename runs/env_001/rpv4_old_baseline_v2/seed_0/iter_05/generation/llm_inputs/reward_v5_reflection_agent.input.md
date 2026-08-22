# 1. Search objective
- target_score: 200.000000
- current_score: -114.951866
- gap_to_target: 314.951866

# 2. Current reward program (score: -114.951866)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract relevant observations
    x, y = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    angle = obs[4]
    left_contact, right_contact = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nx_vel, ny_vel = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_left, n_right = next_obs[6], next_obs[7]

    # --- Progress: reward reducing distance to the landing pad ---
    old_dist = abs(x) + abs(y)
    new_dist = abs(nx) + abs(ny)
    progress_reward = 0.5 * (old_dist - new_dist)  # positive when distance shrinks

    # --- Landing quality change (potential difference) ---
    # Stability factor: 1 when velocity and angle are near zero
    def stability(vx, vy, angle):
        return 1.0 / (1.0 + abs(vx) + abs(vy) + 5.0 * abs(angle))

    cur_stab = stability(x_vel, y_vel, angle)
    next_stab = stability(nx_vel, ny_vel, n_angle)
    cur_contact = left_contact + right_contact
    next_contact = n_left + n_right

    # Weight for landing potential: 2.0
    landing_change = 2.0 * (next_contact * next_stab - cur_contact * cur_stab)

    # --- Velocity change (penalty for speeding up, reward for slowing down) ---
    vel_change = -0.05 * ((abs(nx_vel) + abs(ny_vel)) - (abs(x_vel) + abs(y_vel)))

    # --- Angle change (penalty for increasing tilt, reward for straightening) ---
    ang_change = -0.1 * (abs(n_angle) - abs(angle))

    # --- Fuel efficiency penalty (only when engine is used) ---
    fuel_penalty = -0.02 * (1.0 if action != 0 else 0.0)

    total_reward = progress_reward + landing_change + vel_change + ang_change + fuel_penalty

    components = {
        "progress_reward": progress_reward,
        "landing_change": landing_change,
        "vel_change": vel_change,
        "ang_change": ang_change,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-114.951866, len=68.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.287477, -95.462869]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_change | 2.100201 | 70.2% | 70.6% | 3.1% |
| progress_reward | 0.540807 | 18.1% | 20.3% | 100.0% |
| vel_change | -0.002102 | -0.1% | 5.4% | 100.0% |
| fuel_penalty | -0.089000 | -3.0% | 3.0% | 6.5% |
| ang_change | -0.006531 | -0.2% | 0.7% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
| pct | score_mean | score_std |
|---|---|---|
| 40% | 0.00 | 0.00 |
| 80% | 0.00 | 0.00 |

# 4. Environment facts
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