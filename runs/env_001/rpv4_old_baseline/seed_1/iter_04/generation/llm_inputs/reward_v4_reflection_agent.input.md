# 1. Search objective
- target_score: 200.000000
- current_score: -185.719849
- gap_to_target: 385.719849

# 2. Current reward program (score: -185.719849)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- unpack observations ----
    x_old, y_old = obs[0], obs[1]
    x_new, y_new = next_obs[0], next_obs[1]
    x_vel_new, y_vel_new = next_obs[2], next_obs[3]
    body_angle_new = next_obs[4]
    ang_vel_new = next_obs[5]
    left_contact_new = next_obs[6]
    right_contact_new = next_obs[7]

    # 1. Approach reward – milder to avoid reckless speeding
    dist_old = (x_old**2 + y_old**2) ** 0.5
    dist_new = (x_new**2 + y_new**2) ** 0.5
    approach_reward = 10.0 * (dist_old - dist_new)

    # 2. Altitude reward – encourage reaching the landing pad height (y≈0)
    altitude_reward = 10.0 / (1.0 + 5.0 * abs(y_new))

    # 3. Velocity penalty – baseline penalty, amplified near ground to force a soft landing
    speed = abs(x_vel_new) + abs(y_vel_new)
    near_ground_scale = 3.0 / (abs(y_new) + 0.1)
    vel_penalty = -speed * (1.0 + near_ground_scale)

    # 4. Stability penalty – moderate tilt and angular velocity cost
    stability_penalty = -10.0 * (body_angle_new**2) - 1.0 * (ang_vel_new**2)

    # 5. Thrust cost – small penalty for any engine use to promote fuel efficiency
    thrust_cost = -0.03 if action != 0 else 0.0

    # 6. Soft landing bonus – large reward for gentle two-leg touchdown
    contact_both = 1.0 if (left_contact_new > 0.5 and right_contact_new > 0.5) else 0.0
    if contact_both > 0.5:
        vel_sum = abs(x_vel_new) + abs(y_vel_new)
        angle_abs = abs(body_angle_new)
        vel_factor = 1.0 / (1.0 + 10.0 * vel_sum)
        angle_factor = 1.0 / (1.0 + 5.0 * angle_abs)
        landing_bonus = 500.0 * vel_factor * angle_factor
    else:
        landing_bonus = 0.0

    # ---- assemble ----
    total_reward = (approach_reward +
                    altitude_reward +
                    vel_penalty +
                    stability_penalty +
                    thrust_cost +
                    landing_bonus)
    components = {
        'approach_reward': approach_reward,
        'altitude_reward': altitude_reward,
        'vel_penalty': vel_penalty,
        'stability_penalty': stability_penalty,
        'thrust_cost': thrust_cost,
        'landing_bonus': landing_bonus
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-185.719849, len=576.150000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[-232.937948, -112.939579]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| altitude_reward | 778.118141 | 61.5% | 61.5% | 100.0% |
| vel_penalty | -434.385011 | -34.4% | 34.4% | 100.0% |
| stability_penalty | -29.978593 | -2.4% | 2.4% | 100.0% |
| thrust_cost | -17.250000 | -1.4% | 1.4% | 99.8% |
| approach_reward | -2.254687 | -0.2% | 0.4% | 100.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 4. Environment facts
## 1. 任务目标
主体为一个2D飞行器（vehicle-like），初始位置在视口顶部中央附近，带有随机初始作用力。  
核心目标是 **尽快到达并稳定停靠在中央目标着陆平台上**，同时尽可能少使用引擎推力。  
智能体需要学习：趋近目标、降低线速度、保持姿态竖直、实现两腿同时安全接触（软着陆）。  
不应混淆的目标：单纯求快但忽略安全着陆，或一味省燃料而无法到达目标。

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: [8]  
- **dtype**: 默认为 float64 或 float32（取决于环境实现，但通常为 float64）  
- **obs[0]**: `x_position`，水平方向相对于目标着陆平台中心的偏移量，可用于奖励趋近目标，reward_usable: true  
- **obs[1]**: `y_position`，垂直方向相对于平台高度（接触面）的偏移量，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle`，机体倾角（如弧度），reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，reward_usable: true  
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **动作/索引 0**: `no_engine` (不做任何事)，语义：无推力，用于滑行或停靠后保持  
- **动作/索引 1**: `left_orientation_engine` (左姿态引擎)，语义：产生逆时针或顺时针旋转力矩（具体方向取决于环境）  
- **动作/索引 2**: `main_engine` (主引擎)，语义：产生纵向（向上）推力  
- **动作/索引 3**: `right_orientation_engine` (右姿态引擎)，语义：产生与左引擎相反的旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 可能表示机体静止稳定，通常意味着已着陆并静止，可能视为成功；  
- **failure-like termination**: `crash_or_body_contact` （如机身碰撞地面或平台以外部分）、`horizontal_position_outside_viewport` （漂出水平边界）很可能表示失败；  
- **ambiguous termination**: `crash_or_body_contact` 若接触平台但判定为 crash 则为失败，但描述未区分成功接触与失败接触；我们需要从“到达并稳定停靠”推断理想行为是两腿接触且低速，但不能直接从终止信号中获知成功。  
- **truncation**: 源代码未显示截断（max_steps），但多数环境有步数限制，此处未给出，视为不存在或不可直接用于奖励。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空字典 `{}`，无可用字段 。  
- **forbidden_or_uncertain_info_fields**: 任何 info 字段均不可用（因为提供的信息为空）。终止原因也不能从 info 获取。

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（可直接组成距离度量）  
- **velocity**: `x_velocity`, `y_velocity`（线速度）  
- **orientation**: `body_angle`, `angular_velocity`（角速度）  
- **contact**: `left_support_contact`, `right_support_contact`（双腿触地标志），可构造着陆条件  
- **action/engine**: 可以基于动作选择构建惩罚（如鼓励 no_engine，惩罚主引擎和姿态引擎）  
- **other**: 无