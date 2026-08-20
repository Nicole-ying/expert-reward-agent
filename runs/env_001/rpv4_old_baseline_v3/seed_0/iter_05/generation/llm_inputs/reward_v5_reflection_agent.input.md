# 1. Search objective
- target_score: 200.000000
- current_score: -119.410807
- gap_to_target: 319.410807

# 2. Current reward program (score: -119.410807)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: potential-based approach shaping as primary driver,
    one-shot landing success bonus for terminal incentive, proximity-gated
    velocity damping, and supporting contact/landing-quality signals.
    No continuous proximity reward — prevents hovering exploit.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_left = obs[6]
    prev_right = obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # 1. Potential-based approach shaping — primary driver toward the pad.
    # Bounded: total per episode = w_approach * initial_distance. Cannot be exploited.
    w_approach = 10.0
    approach_reward = w_approach * (dist_prev - dist)

    # 2. One-shot landing success bonus.
    # Fires only on the transition from not-landed to landed, near the pad,
    # with low speed and upright attitude. Gives a large terminal incentive
    # that exceeds any possible cumulative reward from hovering.
    prev_both = prev_left * prev_right
    now_both = left_contact * right_contact
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 0.5)
    soft_landing = float(speed < 1.5)
    upright = float(abs(angle) < 0.5)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * soft_landing * upright

    # 3. Landing quality — continuous reward while settled on both feet.
    # Rewards stability: low speed + upright + both feet down.
    w_landing = 5.0
    landing_quality = w_landing * (left_contact * right_contact) / (1.0 + abs(angle) + speed)

    # 4. Proximity-gated velocity damping.
    # Strong near the pad (enforces soft landing), negligible far away.
    near_gate = 1.0 / (1.0 + 5.0 * dist)
    w_vel = 0.5
    vel_penalty = -w_vel * near_gate * speed

    # 5. Orientation cost — keep the lander upright.
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # 6. Per-foot contact bonus — encourage ground contact.
    w_contact = 1.0
    contact_bonus = w_contact * (left_contact + right_contact)

    # 7. Fuel efficiency — light pressure against unnecessary engine use.
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + success_bonus + landing_quality +
        vel_penalty + orientation_cost + contact_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "success_bonus": success_bonus,
        "landing_quality": landing_quality,
        "vel_penalty": vel_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-119.410807, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.241873, -95.638888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 80.000000 | 73.0% | 73.0% | 0.6% |
| approach_reward | 11.159000 | 10.2% | 10.5% | 100.0% |
| vel_penalty | -8.656974 | -7.9% | 7.9% | 100.0% |
| contact_bonus | 3.300000 | 3.0% | 3.0% | 3.4% |
| orientation_cost | -3.182964 | -2.9% | 2.9% | 100.0% |
| landing_quality | 2.820021 | 2.6% | 2.6% | 1.5% |
| fuel_cost | -0.115500 | -0.1% | 0.1% | 5.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| -111.8 | -111.7 | -119.3 | -123.7 | -117.0 | -118.2 | -115.7 | -119.8 | -119.4 | -120.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 11.203 | 11.196 | 11.169 | 11.133 | 11.174 | 11.169 | 11.179 | 11.162 | 11.166 | 11.164 |
| contact_bonus | 3.050 | 3.250 | 3.100 | 3.200 | 3.050 | 2.950 | 3.050 | 3.150 | 3.000 | 3.200 |
| fuel_cost | -0.261 | -0.198 | -0.621 | -0.603 | -0.171 | -0.170 | -0.149 | -0.261 | -0.145 | -0.123 |
| landing_quality | 3.057 | 3.480 | 3.104 | 2.364 | 3.149 | 3.071 | 3.193 | 3.090 | 3.100 | 2.823 |
| orientation_cost | -1.755 | -1.855 | -2.241 | -3.980 | -2.573 | -2.813 | -1.622 | -2.518 | -2.665 | -3.227 |
| success_bonus | 60.000 | 60.000 | 70.000 | 50.000 | 80.000 | 70.000 | 80.000 | 80.000 | 80.000 | 80.000 |
| vel_penalty | -8.629 | -8.651 | -8.573 | -8.615 | -8.598 | -8.594 | -8.575 | -8.608 | -8.584 | -8.644 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_bonus | 2.9% | 3.0% | 3.1% | 3.4% | 3.1% | 2.9% | 3.0% | 3.1% | 3.0% | 3.2% |
| fuel_cost | 12.7% | 9.6% | 30.3% | 29.4% | 8.3% | 8.3% | 7.2% | 12.7% | 7.1% | 6.0% |
| landing_quality | 1.5% | 1.8% | 1.4% | 1.2% | 1.4% | 1.4% | 1.5% | 1.5% | 1.4% | 1.5% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| success_bonus | 0.4% | 0.4% | 0.5% | 0.4% | 0.6% | 0.5% | 0.6% | 0.6% | 0.6% | 0.6% |
| vel_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

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