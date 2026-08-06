# 1. Search objective
- target_score: 200.000000
- current_score: -10.112911
- gap_to_target: 210.112911

# 2. Current reward program (score: -10.112911)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current and next state
    old_x, old_y = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- Approach reward: distance-based shaping toward target (0,0) ----
    old_dist = (old_x ** 2 + old_y ** 2) ** 0.5
    new_dist = (x ** 2 + y ** 2) ** 0.5
    approach_reward = (old_dist - new_dist) * 5.0   # positive when moving closer

    # ---- Smoothness penalty: discourage high speed and rotation ----
    speed_penalty = -0.2 * (vx ** 2 + vy ** 2) - 0.1 * (angle ** 2 + angvel ** 2)

    # ---- Landing reward: both legs on platform and vehicle stable ----
    if left_contact > 0.5 and right_contact > 0.5:
        # Quality of touchdown: near upright, negligible velocity
        landing_quality = 10.0 - 15.0 * angle ** 2 - 3.0 * vx ** 2 - 3.0 * vy ** 2 - 3.0 * angvel ** 2
        landing_reward = max(0.0, landing_quality)
    else:
        landing_reward = 0.0

    # ---- Fuel penalty: discourage unnecessary engine use ----
    fuel_penalty = -0.05 if action in [1, 2, 3] else 0.0

    # ---- Small per-step penalty to prevent lingering ----
    time_penalty = -0.01

    total = approach_reward + speed_penalty + landing_reward + fuel_penalty + time_penalty

    components = {
        "approach_reward": approach_reward,
        "speed_penalty": speed_penalty,
        "landing_reward": landing_reward,
        "fuel_penalty": fuel_penalty,
        "time_penalty": time_penalty
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-10.112911, len=693.850000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-169.623516, 205.900426]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 173.607873 | 79.7% | 79.7% | 2.6% |
| fuel_penalty | -25.070000 | -11.5% | 11.5% | 72.3% |
| time_penalty | -6.938500 | -3.2% | 3.2% | 100.0% |
| speed_penalty | -6.258816 | -2.9% | 2.9% | 100.0% |
| approach_reward | 4.244203 | 1.9% | 2.7% | 99.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Component checkpoint trajectories
# Reward Component Training Statistics

- steps_seen: 1003520
- reward_error_count_max: 0

| name | mean | abs_mean | nonzero_rate | mean_when_active | abs_mean_when_active | min | max | count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| component.approach_reward | 0.019592 | 0.022193 | 0.999759 | 0.019596 | 0.022198 | -0.146468 | 0.184306 | 1003520 |
| component.fuel_penalty | -0.036596 | 0.036596 | 0.731923 | -0.050000 | 0.050000 | -0.050000 | 0.000000 | 1003520 |
| component.landing_reward | 4.192138 | 4.192138 | 0.437329 | 9.585786 | 9.585786 | 0.000000 | 10.000000 | 1003520 |
| component.speed_penalty | -0.054489 | 0.054489 | 1.000000 | -0.054489 | 0.054489 | -6.192150 | -0.000000 | 1003520 |
| component.time_penalty | -0.010000 | 0.010000 | 1.000000 | -0.010000 | 0.010000 | -0.010000 | -0.010000 | 1003520 |
| component.total_reward | 4.110645 | 4.230719 | 1.000000 | 4.110645 | 4.230719 | -6.305537 | 9.990196 | 1003520 |
| generated_reward | 4.110645 | 4.230719 | 1.000000 | 4.110645 | 4.230719 | -6.305537 | 9.990196 | 1003520 |
| original_env_reward | -0.114082 | 1.848801 | 1.000000 | -0.114082 | 1.848801 | -100.000000 | 132.917108 | 1003520 |

## Per-episode component sums

| component | mean | abs_mean | min | max | episodes |
|---|---:|---:|---:|---:|---:|
| approach_reward | 5.322066 | 5.324719 | -1.987946 | 7.098551 | 3690 |
| fuel_penalty | -9.938415 | 9.938415 | -45.550000 | -1.050000 | 3690 |
| landing_reward | 1138.530665 | 1138.530665 | 0.000000 | 8482.589232 | 3690 |
| speed_penalty | -14.807231 | 14.807231 | -72.087005 | -3.732098 | 3690 |
| time_penalty | -2.715873 | 2.715873 | -10.000000 | -0.540000 | 3690 |
| total_reward | 1116.391213 | 1132.472951 | -74.597186 | 8438.393726 | 3690 |


# 4. Environment facts
## 1. 任务目标
主目标：控制一个从画面顶部中央附近出发的飞行器，安全、稳定地降落在画面中央的目标平台上。要求着陆时速度接近于零、姿态接近竖直，且所有支脚平稳接触平台。

次要目标：在确保主目标达成的前提下，尽量缩短飞行时间，并尽量减少主引擎和姿态引擎的使用（即节省燃料）。

不可混淆的目标：不应将“快速到达”或“节省燃料”凌驾于“安全着陆”之上；也不能将“悬停”或“保持在目标上方”当作成功条件。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测）
- obs[0]: x_position (相对目标平台的水平坐标), 可直接用于距离/接近奖励，reward_usable: true
- obs[1]: y_position (相对目标平台高度的垂直坐标), 同上，reward_usable: true
- obs[2]: x_velocity (水平线速度), 可用于着陆软度控制，reward_usable: true
- obs[3]: y_velocity (垂直线速度), 同上，reward_usable: true
- obs[4]: body_angle (机体朝向角), 可用于姿态奖励，reward_usable: true
- obs[5]: angular_velocity (角速度), 可用于姿态稳定性惩罚，reward_usable: true
- obs[6]: left_support_contact (左侧支脚接触标志，1.0 表示接触), 可用于着陆状态判断，reward_usable: true
- obs[7]: right_support_contact (右侧支脚接触标志，1.0 表示接触), 同上，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine (无推力，仅受重力/物理影响)
- action 1: left_orientation_engine (点燃左侧姿态推进器，产生旋转力矩)
- action 2: main_engine (点燃主推进器，提供向上推力并可能产生力矩)
- action 3: right_orientation_engine (点燃右侧姿态推进器，产生反方向旋转力矩)

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: body_not_awake_or_settled 如果发生在飞行器已接触地面且速度/角速度极低时，极可能意味着成功着陆；但如果发生在半空中或刚碰撞后，则可能是早期终止。
- failure‑like termination: crash_or_body_contact（与地面或障碍的异常碰撞）、horizontal_position_outside_viewport（水平飞出边界）明确为失败。
- ambiguous termination: body_not_awake_or_settled 本身不区分成功/失败，需要结合观察判断。
- truncation: 代码中未出现 episode length 截断，但实际部署时可能通过外部 wrapper 实现，当前源中未见。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info = {} 或未返回任何键）
- forbidden_or_uncertain_info_fields: 任何未在以上列出的字段均不可用

## 7. 可用于奖励函数的信号
- position: x_position, y_position（可直接计算到目标的距离、高度差）
- velocity: x_velocity, y_velocity（可衡量接近速度、着陆软度）
- orientation: body_angle（偏离竖直的角度），angular_velocity（旋转速度）
- contact: left_support_contact, right_support_contact（着陆脚是否触地，可判断着陆状态）
- action/engine: action 索引可映射到是否使用主引擎、姿态引擎，用于推力/燃料惩罚
- other: 可通过 (obs, next_obs) 的组合构造微分信号，如速度变化、角速度变化等