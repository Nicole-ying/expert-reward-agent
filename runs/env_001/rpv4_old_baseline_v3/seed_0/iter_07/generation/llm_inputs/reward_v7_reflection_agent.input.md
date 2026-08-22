# 1. Search objective
- target_score: 200.000000
- current_score: -114.278961
- gap_to_target: 314.278961

# 2. Current reward program (score: -114.278961)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Two-tier speed penalty + strong approach shaping for 2D lander.

    Replaces the failed speed-corridor with:
    - Gentle universal speed cost (encourages efficient descent)
    - Proximity-gated near-ground damping (enforces soft landing)
    Combined with a boosted potential-based approach reward as the primary
    descent driver, and a large terminal success bonus.
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ------------------------------------------------------------------
    # 1. Potential-based approach shaping — PRIMARY descent driver.
    #    Boosted from 5.0 to 8.0 so it can overcome penalty noise.
    #    Total per episode capped at w_app * initial_distance (unexploitable).
    # ------------------------------------------------------------------
    w_app = 8.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Gentle universal speed cost — discourages reckless speed.
    #    Same form as the successful iteration 2 (score +40).
    #    At speed=2.0 this is only -0.2/step — does not dominate.
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 3. Proximity-gated near-ground speed damping.
    #    gate = 1/(1+2*dist): at dist=0 gate=1.0, at dist=2 gate=0.2,
    #    at dist=5 gate=0.09.  Strong near pad, negligible far away.
    #    This replaces the speed corridor with a one-sided penalty that
    #    only asks the agent to slow down when it matters.
    # ------------------------------------------------------------------
    near_gate = 1.0 / (1.0 + 2.0 * dist)
    w_near = 1.0
    near_speed_penalty = -w_near * near_gate * speed

    # ------------------------------------------------------------------
    # 4. Horizontal drift penalty — always undesirable.
    # ------------------------------------------------------------------
    w_drift = 0.4
    drift_penalty = -w_drift * abs(vx)

    # ------------------------------------------------------------------
    # 5. Orientation cost — keep the lander upright.
    # ------------------------------------------------------------------
    w_angle = 0.8
    w_angvel = 0.3
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # ------------------------------------------------------------------
    # 6. Per-foot contact bonus — encourage ground contact.
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 7. Landing quality — continuous bonus while both feet are down.
    # ------------------------------------------------------------------
    both = left * right
    w_land = 6.0
    landing_quality = w_land * both / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 8. Terminal success bonus — one-shot, quality-scaled.
    #    Wider near_pad (1.5 vs old 1.2) and bigger weight (200 vs 120).
    #    Makes landing discovery easier and more rewarding.
    # ------------------------------------------------------------------
    prev_both = obs[6] * obs[7]
    now_both = left * right
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 1.5)
    land_quality = 1.0 / (1.0 + abs(angle) + speed)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * land_quality

    # ------------------------------------------------------------------
    # 9. Fuel efficiency — light pressure against unnecessary engine use.
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + vel_cost + near_speed_penalty +
        drift_penalty + orientation_cost + contact_bonus +
        landing_quality + success_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "vel_cost": vel_cost,
        "near_speed_penalty": near_speed_penalty,
        "drift_penalty": drift_penalty,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "success_bonus": success_bonus,
        "fuel_cost": fuel_cost,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-114.278961, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-139.681272, -90.479014]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 77.198031 | 51.8% | 51.8% | 1.2% |
| near_speed_penalty | -30.589482 | -20.5% | 20.5% | 100.0% |
| drift_penalty | -11.271814 | -7.6% | 7.6% | 100.0% |
| approach_reward | 8.945437 | 6.0% | 6.2% | 100.0% |
| vel_cost | -7.325735 | -4.9% | 4.9% | 100.0% |
| contact_bonus | 6.400000 | 4.3% | 4.3% | 3.1% |
| landing_quality | 3.609407 | 2.4% | 2.4% | 1.5% |
| orientation_cost | -3.393472 | -2.3% | 2.3% | 100.0% |
| fuel_cost | -0.106500 | -0.1% | 0.1% | 5.2% |

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
| -116.4 | -120.2 | -109.1 | -115.9 | -119.5 | -116.2 | -116.2 | -115.7 | -113.1 | -114.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 9.007 | 8.928 | 8.957 | 8.932 | 8.930 | 8.939 | 8.932 | 8.938 | 8.947 | 8.941 |
| contact_bonus | 8.500 | 6.200 | 6.800 | 6.600 | 6.100 | 6.300 | 6.600 | 6.400 | 6.600 | 6.500 |
| drift_penalty | -12.035 | -11.410 | -11.200 | -11.389 | -11.386 | -11.311 | -11.371 | -11.300 | -11.256 | -11.297 |
| fuel_cost | -0.116 | -0.138 | -0.281 | -0.147 | -0.137 | -0.115 | -0.121 | -0.112 | -0.121 | -0.105 |
| landing_quality | 3.563 | 3.644 | 4.524 | 3.671 | 3.302 | 3.717 | 4.006 | 3.807 | 4.293 | 3.670 |
| near_speed_penalty | -31.142 | -30.483 | -30.501 | -30.583 | -30.497 | -30.490 | -30.451 | -30.448 | -30.582 | -30.581 |
| orientation_cost | -11.624 | -7.328 | -2.801 | -6.444 | -6.349 | -4.435 | -5.711 | -4.638 | -3.198 | -3.525 |
| success_bonus | 64.545 | 67.249 | 68.494 | 58.731 | 75.087 | 79.019 | 90.398 | 86.803 | 80.467 | 77.531 |
| vel_cost | -7.464 | -7.334 | -7.311 | -7.345 | -7.329 | -7.318 | -7.321 | -7.312 | -7.324 | -7.328 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_bonus | 4.5% | 3.1% | 3.2% | 3.3% | 3.1% | 3.1% | 3.4% | 3.2% | 3.1% | 3.2% |
| drift_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_cost | 5.5% | 6.7% | 13.7% | 7.2% | 6.7% | 5.6% | 5.9% | 5.5% | 5.9% | 5.1% |
| landing_quality | 1.6% | 1.5% | 1.8% | 1.5% | 1.4% | 1.5% | 1.5% | 1.5% | 1.8% | 1.5% |
| near_speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| success_bonus | 1.0% | 1.0% | 1.1% | 1.0% | 1.1% | 1.2% | 1.1% | 1.1% | 1.2% | 1.2% |
| vel_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

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