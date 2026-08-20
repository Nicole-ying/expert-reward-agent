# 1. Search objective
- target_score: 200.000000
- current_score: -45.143561
- gap_to_target: 245.143561

# 2. Current reward program (score: -45.143561)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: exponential proximity attraction as primary signal,
    strong potential-based approach shaping, proximity-gated velocity damping,
    relaxed orientation penalty, and moderate contact/landing bonuses.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. Exponential proximity: primary continuous attractor toward pad center
    # Range: ~1.5 at pad (dist=0), ~0.55 at dist=2.5, ~0.07 at dist=7.5, ~0.0005 at dist=20
    w_proximity = 1.5
    dist_scale = 2.5
    proximity_reward = w_proximity * (2.718281828 ** (-dist / dist_scale))

    # 2. Potential-based approach: strong directional shaping
    # Rewards every step that reduces distance to pad
    w_approach = 20.0
    approach_reward = w_approach * (dist_prev - dist)

    # 3. Proximity-gated velocity damping: penalise speed only near the pad
    # Far away: allow free movement for efficient descent. Near pad: enforce soft landing.
    near_pad = 2.718281828 ** (-dist / 1.5)
    w_vel_near = 1.0
    w_vel_far = 0.02
    vel_penalty = -(w_vel_near * near_pad + w_vel_far) * (abs(vx) + abs(vy))

    # 4. Orientation cost: relaxed to allow necessary tilting for horizontal control
    w_angle = 0.3
    w_angvel = 0.05
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 5. Per-foot contact bonus: moderate reward for ground contact
    w_contact = 0.5
    contact_bonus = w_contact * (left_contact + right_contact)

    # 6. Landing quality: bonus for stable two-foot contact
    both_feet = left_contact * right_contact
    w_landing = 2.0
    landing_quality = w_landing * both_feet / (1.0 + abs(angle) + speed)

    # 7. Fuel efficiency: light penalty for engine use
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity_reward + vel_penalty +
        orientation_cost + contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity_reward": proximity_reward,
        "vel_penalty": vel_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-45.143561, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-76.943309, -0.360620]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_reward | 1175.046021 | 89.0% | 89.0% | 100.0% |
| vel_penalty | -86.206726 | -6.5% | 6.5% | 100.0% |
| fuel_cost | -29.739000 | -2.3% | 2.3% | 99.1% |
| approach_reward | 18.935215 | 1.4% | 1.8% | 100.0% |
| landing_quality | 2.535736 | 0.2% | 0.2% | 0.2% |
| contact_bonus | 2.025000 | 0.2% | 0.2% | 0.2% |
| orientation_cost | -0.943085 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 3.5. Checkpoint evaluations (every 10% of training)
# Checkpoint 评估（训练过程中各组件的变化趋势）

每个 checkpoint 在训练到 N% 时用当前模型评估 20 集，记录每个组件的 ep_sum_mean。
**趋势比单点数值更重要：**
- 组件全程为 0 → 死组件，agent 在训练中忽略了它
- 组件从 0 逐渐增长 → 正在引导学习
- 组件骤升 → 可能被 exploit
- 组件骤降 → agent 在学习抑制它

## Overall score trend
| cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---|---|---|---|---|---|---|---|---|---|---|
| -936.8 | -606.8 | -230.3 | -135.4 | -64.9 | -68.2 | -47.1 | 160.3 | -16.2 | -29.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | -130.979 | -89.959 | -16.776 | 1.130 | 14.842 | 12.223 | 17.675 | 24.117 | 24.801 | 20.572 |
| contact_bonus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 90.125 | 0.000 | 7.275 |
| fuel_cost | -8.901 | -29.135 | -10.410 | -30.000 | -29.964 | -18.721 | -25.745 | -14.955 | -29.994 | -29.343 |
| landing_quality | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 142.385 | 0.000 | 7.555 |
| orientation_cost | -0.994 | -0.748 | -0.724 | -0.707 | -0.910 | -1.074 | -0.923 | -1.220 | -0.967 | -1.519 |
| proximity_reward | 119.220 | 386.341 | 249.979 | 883.271 | 1062.689 | 1017.000 | 1109.791 | 687.213 | 1262.633 | 1181.224 |
| vel_penalty | -49.655 | -48.468 | -47.537 | -54.515 | -62.950 | -52.407 | -66.052 | -76.554 | -97.525 | -90.322 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 98.2% | 100.0% | 100.0% |
| contact_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 18.0% | 0.0% | 0.9% |
| fuel_cost | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 62.4% | 85.8% | 90.5% | 100.0% | 99.9% |
| landing_quality | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 14.7% | 0.0% | 0.6% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% |

# 4. Environment facts
## 1. 任务目标
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (实际为连续值，左/右支撑为 0.0 或 1.0)
- 各维度含义：
  - obs[0] (x_position): 相对于目标着陆垫的水平坐标，reward_usable: true  
  - obs[1] (y_position): 相对于着陆垫高度的垂直坐标，reward_usable: true  
  - obs[2] (x_velocity): 水平线速度，reward_usable: true  
  - obs[3] (y_velocity): 垂直线速度，reward_usable: true  
  - obs[4] (body_angle): 机体角度（方向），reward_usable: true  
  - obs[5] (angular_velocity): 角速度，reward_usable: true  
  - obs[6] (left_support_contact): 左侧支撑接触标志（1.0表示接触，0.0表示未接触），reward_usable: true  
  - obs[7] (right_support_contact): 右侧支撑接触标志，reward_usable: true

所有观测维度均可在自定义奖励中安全使用。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine – 什么都不做，滑翔/惯性飞行
  - action 1: left_orientation_engine – 点燃左侧姿态发动机（产生逆时针或顺时针旋转力矩）
  - action 2: main_engine – 点燃主发动机（提供向机头方向的推力，通常向上或减速）
  - action 3: right_orientation_engine – 点燃右侧姿态发动机（与左侧相反的旋转力矩）

注意：该飞行器没有横向直接推进器，水平控制主要通过倾斜机身+主发动机实现。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 中的 “settled” 部分 —— 推测为飞行器稳定停靠在目标着陆垫上（满足速度、角度阈值且支撑接触）
- failure-like termination: crash_or_body_contact（当不属于成功着陆时的身体/地面碰撞）和 horizontal_position_outside_viewport
- ambiguous termination: body_not_awake_or_settled 内部 —— 由于任务描述未区分“settled”与“not awake”，但根据任务目标“reach and settle”可推断出成功情况；需结合接触标志、位置接近度进一步解析
- truncation: 无（step 返回的 truncated 恒为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，终止后无明确标志）
- explicit_failure_flag_available: false 
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均未提供，不可依赖。

成功/失败只能通过下一状态的观测（next_obs）与 done 组合推断：例如，若终止时左右支撑接触均为 1，且 x、y 位置接近 0，速度很小，角度接近 0（或垂直），则可以假定为成功着陆。

## 7. 可用于奖励函数的信号
- 位置信号：
  - next_obs[0] (x_position): 相对于目标着陆垫的水平偏差
  - next_obs[1] (y_position): 相对于着陆垫高度的垂直偏差（越高越正？）
- 速度信号：
  - next_obs[2] (x_velocity): 水平速度（靠近着陆垫时应接近 0）
  - next_obs[3] (y_velocity): 垂直速度（靠近着陆垫时应接近 0 或一个小负值以表示软下降）
- 姿态信号：
  - next_obs[4] (body_angle): 机体角度，用于维持稳定姿态（接近 0 表示竖直）
  - next_obs[5] (angular_velocity): 角速度，用于惩罚大幅旋转
- 接触信号：
  - next_obs[6] (left_support_contact): 左侧支撑接触 (0/1)
  - next_obs[7] (right_support_contact): 右侧支撑接触 (0/1)
  可用于判断着陆状态、两侧接触平衡、软/硬着陆
- 动作/发动机信号：
  - action: 离散动作，可用于惩罚推力使用（action 0 为无推力，action 1/3 为姿态推力，action 2 为主发动机推力）
- 其他衍生物：
  - 距离到目标：sqrt(x_position² + y_position²)
  - 接近速度/方向
  - 着陆瞬间的状态组合（位置、速度、角度）