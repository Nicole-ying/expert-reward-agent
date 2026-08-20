# Search objective
- target_score: 200.000000
- current_score: 200.987626
- gap_to_target: -0.987626
- target_achievement_ratio: 100.494%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 200.987626）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # weights and sensitivities
    w_progress = 1.0
    w_angle_penalty = 0.2
    w_landing = 1.0
    w_approach_vel = 0.3   # proximity-gated velocity penalty near target
    a_v = 10.0             # sensitivity for vertical speed in landing quality
    b_angle = 10.0         # sensitivity for body angle in landing quality

    # current and next distances to target (0,0)
    dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 1. Main progress signal: improvement towards target
    progress = w_progress * (dist - next_dist)

    # 2. Orientation stability: penalty for body angle to prevent tumbling
    orientation_penalty = -w_angle_penalty * (next_obs[4] ** 2)

    # 3. Landing quality bonus: soft success proxy when leg contacts are active
    contact = max(next_obs[6], next_obs[7])  # any leg contact
    vertical_speed_factor = 1.0 / (1.0 + a_v * (next_obs[3] ** 2))
    angle_factor = 1.0 / (1.0 + b_angle * (next_obs[4] ** 2))
    landing_quality = w_landing * contact * vertical_speed_factor * angle_factor

    # 4. Approach-phase velocity penalty: penalize high speed near target
    #    Proximity gates the penalty: negligible when far, active when close
    proximity = 1.0 / (1.0 + next_dist)
    velocity_sq = next_obs[2] ** 2 + next_obs[3] ** 2
    approach_velocity_penalty = -w_approach_vel * proximity * velocity_sq

    total_reward = progress + orientation_penalty + landing_quality + approach_velocity_penalty

    components = {
        "progress": progress,
        "orientation_penalty": orientation_penalty,
        "landing_quality": landing_quality,
        "approach_velocity_penalty": approach_velocity_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=200.987626, len=626.050000, terminated=13/20, truncated=7/20, reward_errors=0
score_range=[54.033136, 275.481878]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 316.337239 | 97.8% | 97.8% | 51.0% |
| approach_velocity_penalty | -5.152263 | -1.6% | 1.6% | 99.9% |
| progress | 1.302700 | 0.4% | 0.4% | 100.0% |
| orientation_penalty | -0.678321 | -0.2% | 0.2% | 99.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 飞行器软着陆任务。目标是从初始位置（接近视图顶部中央，具有随机初始力）飞到中央的目标着陆平台，**平稳着陆并停稳**。任务主要目标是**成功、安全地到达目标并停靠**，附属目标包括**尽快完成**和**尽量节省燃料**（即尽量少用引擎）。智能体需要学会接近目标、减速、保持稳定姿态，并以低速、小角度接触平台。不要把“省燃料”或“快速”当作主目标，它们不能以牺牲着陆安全为代价。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32（默认）  
- 各维度含义与奖励可用性：
  - obs[0]: x_position（相对目标平台的水平偏移），reward_usable: true  
  - obs[1]: y_position（相对平台高度的垂直偏移），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体倾斜角），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支撑腿接触标志，1.0 表示接触，0.0 表示未接触），reward_usable: true  
  - obs[7]: right_support_contact（右支撑腿接触标志），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义：
  - action 0: no_engine（不点火，无推力）  
  - action 1: left_orientation_engine（点燃左姿态引擎，产生侧向力矩或推力）  
  - action 2: main_engine（点燃主引擎，提供主要上升/下降推力）  
  - action 3: right_orientation_engine（点燃右姿态引擎）

## 5. step 与终止条件分析
### 5.1 终止模式
根据环境源码，terminated 由以下条件之一触发：
- crash_or_body_contact：机体发生碰撞或非正常接触（例如翻滚导致机身触地），通常为失败。  
- horizontal_position_outside_viewport：水平位置超出可视边界，视为失控失败。  
- body_not_awake_or_settled：机体不再活跃或已稳定停靠。该终止可能意味着成功（已经停稳在平台上），但也可能因陷入静止失败状态而终止。  

**因此**：
- success-like termination：当 body_not_awake_or_settled 触发，且同时满足 left/right support contact 均为 1、位置靠近中心、速度极低、姿态接近水平。  
- failure-like termination：crash_or_body_contact 或 horizontal_position_outside_viewport 触发；或者虽然 settled 但状态不满足安全着陆条件（如姿态过大、偏离平台）。  
- ambiguous termination：body_not_awake_or_settled 本身需要结合观测状态才能判断是否成功。没有显式的“success”或“failure”标志。  
- truncation：环境中未观察到 episode 截断（步数限制未体现），但假设存在最大步数可能属于 truncation。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: {}（info 为空字典，根本不可用）  
- forbidden_or_uncertain_info_fields: original_reward, 以及任何未在 step 源码中显式返回的 info 字段。

## 7. 可用于奖励函数的信号
- position: x_position, y_position（相对目标平台的偏移）  
- velocity: x_velocity, y_velocity  
- orientation: body_angle, angular_velocity  
- contact: left_support_contact, right_support_contact  
- action/engine: 动作编号（0,1,2,3）  
- other: 无其他显式信号

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2d_lander_with_two_legs
  actuator_type: thrusters (main engine + orientation engines)
  contact_structure: two_point_contacts (left and right legs)
primary_objectives:
  - 平稳降落在目标平台上（left 和 right 腿接触，机体接近水平，速度极小）
  - 避免碰撞、翻滚或飞出视口
secondary_objectives:
  - 尽快完成着陆（时间效率）
  - 最少燃料消耗（尽量不用引擎）
main_failure_risks:
  - 姿态失控导致翻滚或机身直接撞地
  - 水平方向偏离过大飞出边界
  - 着陆时垂直速度过大导致硬着陆
  - 过早关闭引擎导致悬停失败
  - 过度使用引擎浪费燃料或过度调整姿态
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_landing_success
  purpose: 鼓励成功、稳定地着陆在平台上
  why_required: 任务核心；没有它智能体可能不学习着陆，或停留在空中不动
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 只能用接触和位置/速度推断成功，可能存在伪成功（如卡在平台边缘也触发接触）

- role_id: crash_and_out_of_bounds_prevention
  purpose: 惩罚碰撞、翻滚或飞出视口的行为
  why_required: 安全约束；必须让智能体避免致命失败
  usable_signals: [body_angle, x_position, y_position, left_support_contact, right_support_contact]
  risks: 角度阈值和位置边界需谨慎设定，过于严格可能阻止正常机动

- role_id: soft_landing_condition
  purpose: 接触平台时确保低速、姿态平稳
  why_required: 即使接触也会因为速度过快导致失败（被环境检测为 crash）
  usable_signals: [y_velocity, body_angle, left_support_contact, right_support_contact]
  risks: 过早惩罚速度可能阻碍接近目标，必须结合接触标志

- role_id: orientation_stability
  purpose: 保持机体尽量接近水平，防止翻滚
  why_required: 姿态过大易引发 crash，且是成功着陆条件之一
  usable_signals: [body_angle]
  risks: 过度惩罚正常转向可能妨碍姿态调整，需与角速度结合

- role_id: progress_towards_target
  purpose: 在飞行阶段引导智能体向目标平台移动
  why_required: 没有引导可能随机漂浮浪费时间
  usable_signals: [x_position, y_position, x_velocity, y_velocity]
  risks: 仅靠位置奖励可能导致智能体高速冲向平台，必须配合软着陆约束

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 尽量减少引擎使用次数
  condition_to_use: 当智能体已具备基本着陆能力（成功率 > 阈值）后加入，或在训练后期逐步增强
  usable_signals: [action]
  risks: 过早加入会抑制必要的引擎使用，导致无法完成着陆

### 10.3 慎用/禁用职责 avoid_roles
- role_id: time_optimization
  reason: 观察空间中没有时间或步数信号，无法可靠测量时间效率；且容易与燃料节约冲突。
  forbidden_or_missing_signals: [time, step_count]

- role_id: exact_center_landing
  reason: 只要在平台上且稳定就行，无需极致中心对齐；过分强调可能使探索变得困难。
  related_signals: 无，但可用 x_position 过度约束

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_landing_success | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | explicit success flag | dense_state_signal (基于距离和速度的阈值判断) + bounded_signal (当条件满足时给予固定奖励) | 需要区分“在空中”和“已着陆”，使用接触标志与速度阈值结合 |
| crash_and_out_of_bounds_prevention | body_angle, x_position, y_position | explicit failure flag | quadratic_penalty (角度/位置超出安全范围) 或 bounded_signal (一旦超出给予负奖励) | 边界和角度阈值需调参 |
| soft_landing_condition | y_velocity, body_angle, left_support_contact, right_support_contact | 无 | dense_state_signal (接触时垂直速度和角度绝对值惩罚) | 仅在接触时激活 |
| orientation_stability |