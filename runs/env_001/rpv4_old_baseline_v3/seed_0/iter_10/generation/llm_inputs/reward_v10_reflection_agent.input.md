# 1. Search objective
- target_score: 200.000000
- current_score: 35.633589
- gap_to_target: 164.366411

# 2. Current reward program (score: 35.633589)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Iter-2 skeleton restored (proven: score +40, 6/20 landings).
    + controlled-descent incentive (speed-gated to prevent plummeting)
    + boosted contact/landing weights
    - crash_penalty removed (poisons ground-contact exploration)
    - main_decel_bonus removed (rewards hovering, anti-landing)
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
    # 1. Potential-based approach — ITER 2 WEIGHT (proven safe)
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Proximity — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Controlled-descent incentive (NEW — replaces harmful main_decel_bonus)
    #    Rewards descending (y decreasing), gated by speed so fast plummeting
    #    gives much less reward than slow, controlled descent.
    #    At speed=0: full weight; at speed=3: weight/4; at speed=10: weight/11.
    # ------------------------------------------------------------------
    descent = max(0.0, y_prev - y)
    w_desc = 3.0
    descent_reward = w_desc * descent / (1.0 + speed)

    # ------------------------------------------------------------------
    # 4. Velocity cost — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 5. Orientation cost — ITER 2 WEIGHTS
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 6. Per-foot contact bonus — BOOSTED from iter-2's 0.5 → 2.0
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 7. Landing quality — BOOSTED from iter-2's 2.0 → 5.0
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 5.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — ITER 2 WEIGHT
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity + descent_reward + vel_cost +
        orientation_cost + contact_bonus + landing_quality + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "descent_reward": descent_reward,
        "vel_cost": vel_cost,
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
score=35.633589, len=574.250000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[-223.136820, 243.361624]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_quality | 142.827770 | 43.4% | 43.4% | 5.5% |
| contact_bonus | 131.000000 | 39.8% | 39.8% | 5.9% |
| proximity | 19.754417 | 6.0% | 6.0% | 100.0% |
| fuel_cost | -13.909500 | -4.2% | 4.2% | 80.7% |
| vel_cost | -10.891025 | -3.3% | 3.3% | 99.9% |
| approach_reward | 4.179364 | 1.3% | 2.0% | 98.8% |
| descent_reward | 2.974024 | 0.9% | 0.9% | 75.8% |
| orientation_cost | -1.078823 | -0.3% | 0.3% | 100.0% |

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
| -122.8 | -59.2 | -26.0 | -90.5 | 43.1 | -56.3 | -59.9 | -71.5 | -15.3 | 62.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 5.554 | 3.214 | 3.077 | 2.542 | 4.079 | 2.932 | 2.794 | 2.792 | 3.584 | 4.668 |
| contact_bonus | 6.200 | 61.600 | 142.000 | 171.200 | 386.000 | 165.700 | 27.900 | 59.100 | 150.400 | 206.200 |
| descent_reward | 1.905 | 2.286 | 2.670 | 2.665 | 2.747 | 2.750 | 2.750 | 2.664 | 2.952 | 2.931 |
| fuel_cost | -0.178 | -7.387 | -8.868 | -12.318 | -8.002 | -10.525 | -8.745 | -10.767 | -14.427 | -10.204 |
| landing_quality | 2.195 | 23.457 | 74.681 | 144.203 | 297.280 | 178.059 | 15.142 | 31.422 | 122.856 | 205.757 |
| orientation_cost | -3.346 | -1.634 | -4.976 | -2.802 | -2.372 | -1.029 | -1.072 | -2.578 | -2.728 | -1.822 |
| proximity | 1.815 | 9.504 | 13.515 | 17.974 | 13.737 | 17.037 | 13.270 | 15.175 | 19.305 | 15.041 |
| vel_cost | -7.364 | -11.426 | -12.454 | -12.337 | -9.517 | -11.681 | -11.770 | -13.347 | -11.066 | -9.992 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% | 100.0% | 99.6% | 99.8% | 99.6% | 98.8% |
| contact_bonus | 3.5% | 7.8% | 12.0% | 8.3% | 30.4% | 8.0% | 2.4% | 4.5% | 7.9% | 12.5% |
| descent_reward | 94.3% | 65.0% | 80.6% | 76.9% | 76.4% | 79.4% | 82.0% | 75.2% | 74.9% | 85.0% |
| fuel_cost | 8.7% | 80.0% | 73.2% | 72.6% | 67.2% | 66.0% | 69.1% | 77.5% | 82.3% | 76.4% |
| landing_quality | 1.0% | 2.2% | 5.6% | 6.8% | 18.3% | 7.5% | 0.9% | 1.9% | 4.9% | 10.7% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_cost | 100.0% | 100.0% | 99.9% | 100.0% | 99.9% | 100.0% | 100.0% | 100.0% | 99.9% | 99.9% |

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