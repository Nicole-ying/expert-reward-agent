# 1. Search objective
- target_score: 200.000000
- current_score: -63.446611
- gap_to_target: 263.446611

# 2. Current reward program (score: -63.446611)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y, vx, vy, angle, omega, lc, rc = obs
    nx, ny, nvx, nvy, nangle, nomega, nlc, nrc = next_obs

    # Distance to the pad centre
    dist = (nx * nx + ny * ny) ** 0.5
    # Total linear speed
    speed = abs(nvx) + abs(nvy)
    # Absolute tilt angle
    tilt = abs(nangle)

    # 1. Proximity reward: always >= 0, highest when on the pad
    proximity_reward = 1.0 / (1.0 + dist)

    # 2. Speed reward: encourages gentle touchdown
    speed_reward = 1.0 / (1.0 + speed)

    # 3. Orientation reward: favours staying upright
    angle_reward = 1.0 / (1.0 + 5.0 * tilt)

    # 4. Contact bonus: tiny reward for each leg touching the ground
    contact_reward = 0.2 * (nlc + nrc)

    # 5. Fuel penalty: minuscule cost to promote efficiency
    fuel_penalty = -0.01 if action != 0 else 0.0

    # 6. Sparse success bonus – unchanged loose condition from previous function
    success = (
        nlc > 0.5 and nrc > 0.5 and
        ny < 0.3 and
        tilt < 0.3 and
        abs(nvx) < 0.2 and abs(nvy) < 0.2
    )
    success_bonus = 200.0 if success else 0.0

    total = proximity_reward + speed_reward + angle_reward + contact_reward + fuel_penalty + success_bonus

    components = {
        "proximity_reward": proximity_reward,
        "speed_reward": speed_reward,
        "angle_reward": angle_reward,
        "contact_reward": contact_reward,
        "fuel_penalty": fuel_penalty,
        "success_bonus": success_bonus,
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-63.446611, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-93.829840, -28.841668]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| speed_reward | 893.127342 | 38.1% | 38.1% | 100.0% |
| angle_reward | 882.158499 | 37.7% | 37.7% | 100.0% |
| proximity_reward | 556.372235 | 23.8% | 23.8% | 100.0% |
| fuel_penalty | -10.000000 | -0.4% | 0.4% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
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