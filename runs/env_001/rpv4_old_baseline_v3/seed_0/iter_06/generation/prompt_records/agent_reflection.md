# Prompt Record

## System Prompt

```text
你是奖励函数修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- 你将看到：最终评估的组件表（episode_sum_mean, magnitude_share, active_rate），以及训练过程中的组件轨迹（每个 checkpoint 的标量值，含 max/mean/min）。轨迹能告诉你组件是在学习还是停滞了。
- 组件统计是观察证据，不是因果贡献。必须结合 score、episode_length、terminated/truncated 判断。

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。

# 代码约束

- 禁止 terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的 obs、next_obs、action 和 info 字段，不得发明字段、切片维度或新输入。
- 第一个 Python code block 只能包含一个完整的 `compute_reward` 函数；不要写 import、class、try/except 或额外函数，不要使用 self。
- 禁止 eval/exec/open，禁止使用 original_reward 或原始环境 reward。
- 需要平方根时使用 `** 0.5`，禁止 import numpy。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`；components 只放总公式中直接出现的奖励组件。

# 输出

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.

```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -48.745400
- gap_to_target: 248.745400

# 2. Current reward program (score: -48.745400)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Velocity-corridor reward for 2D lander.

    Core idea: instead of a narrow proximity-gated velocity penalty, shape the
    entire descent with a *target-speed corridor*.  Far from the pad, faster
    descent is rewarded; near the pad, only slow speeds are rewarded.  This
    creates a continuous, learnable gradient from "descending" through
    "decelerating" to "soft contact."

    Additional signals: potential-based approach, broadened terminal success,
    horizontal drift penalty, orientation cost, contact/landing quality, fuel.
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
    # 1. Potential-based approach shaping
    #    Bounded total = w_app * initial_distance.  Cannot be exploited by
    #    oscillating because going away gives symmetric negative reward.
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Speed corridor — the central innovation
    #    target_speed(d) = v_max * (1 - exp(-d / d_scale))
    #    At d=0:  target=0       → must be nearly stationary at the pad
    #    At d=2:  target≈1.65    → moderate speed
    #    At d=6:  target≈2.82    → faster descent OK far away
    #    At d=15: target≈2.99    → essentially unconstrained
    #
    #    Penalising |speed - target_speed| teaches *progressive deceleration*
    #    across the whole descent, not just the last 0.5 units.
    # ------------------------------------------------------------------
    v_max = 3.0
    d_scale = 2.5
    target_speed = v_max * (1.0 - 2.718281828 ** (-dist / d_scale))
    speed_error = abs(speed - target_speed)
    w_speed = 1.2
    speed_shaping = -w_speed * speed_error

    # ------------------------------------------------------------------
    # 3. Horizontal drift penalty — always undesirable
    #    Separated from vertical so the corridor only governs descent rate,
    #    while lateral motion is penalised uniformly.
    # ------------------------------------------------------------------
    w_drift = 0.4
    drift_penalty = -w_drift * abs(vx)

    # ------------------------------------------------------------------
    # 4. Orientation cost — keep the lander upright
    # ------------------------------------------------------------------
    w_angle = 0.8
    w_angvel = 0.2
    orientation_cost = -w_angle * abs(angle) - w_angvel * abs(angvel)

    # ------------------------------------------------------------------
    # 5. Per-foot contact bonus — encourage ground contact
    # ------------------------------------------------------------------
    w_contact = 2.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 6. Landing quality — continuous bonus while both feet are down
    # ------------------------------------------------------------------
    both = left * right
    w_land = 5.0
    landing_quality = w_land * both / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 7. Terminal success bonus — broadened and quality-scaled
    #    Fires on the transition to both-feet-down, with a wider near_pad
    #    radius (1.2 instead of 0.5) and continuous quality scaling so the
    #    agent gets partial credit for imperfect landings and can improve.
    # ------------------------------------------------------------------
    prev_both = obs[6] * obs[7]
    now_both = left * right
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 1.2)
    land_quality = 1.0 / (1.0 + abs(angle) + speed)
    w_success = 120.0
    success_bonus = w_success * just_landed * near_pad * land_quality

    # ------------------------------------------------------------------
    # 8. Fuel efficiency
    # ------------------------------------------------------------------
    w_fuel = 0.04
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + speed_shaping + drift_penalty +
        orientation_cost + contact_bonus + landing_quality +
        success_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "speed_shaping": speed_shaping,
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
score=-48.745400, len=101.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-133.405059, 33.235863]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 63.196637 | 28.6% | 28.6% | 0.9% |
| speed_shaping | -58.994000 | -26.7% | 26.7% | 100.0% |
| contact_bonus | 40.100000 | 18.1% | 18.1% | 11.3% |
| landing_quality | 28.104392 | 12.7% | 12.7% | 8.3% |
| drift_penalty | -12.456011 | -5.6% | 5.6% | 100.0% |
| orientation_cost | -10.651687 | -4.8% | 4.8% | 100.0% |
| approach_reward | 5.491142 | 2.5% | 2.6% | 100.0% |
| fuel_cost | -1.858000 | -0.8% | 0.8% | 45.6% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 8/20
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
| -107.1 | -95.1 | -80.6 | -84.3 | -79.6 | -88.1 | -83.9 | -85.8 | -33.9 | -49.2 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 5.606 | 5.570 | 5.376 | 5.388 | 5.287 | 5.401 | 5.407 | 5.324 | 5.393 | 5.584 |
| contact_bonus | 7.000 | 6.800 | 7.500 | 8.100 | 7.900 | 7.100 | 9.100 | 7.400 | 43.400 | 15.700 |
| drift_penalty | -11.122 | -11.425 | -12.934 | -12.798 | -13.008 | -12.747 | -12.653 | -13.012 | -13.122 | -11.973 |
| fuel_cost | -2.698 | -0.346 | -0.770 | -0.766 | -0.892 | -0.630 | -1.000 | -0.760 | -2.074 | -1.554 |
| landing_quality | 3.397 | 3.221 | 3.272 | 3.867 | 4.011 | 3.156 | 4.509 | 3.321 | 27.814 | 8.247 |
| orientation_cost | -3.235 | -3.163 | -6.260 | -4.763 | -4.049 | -4.054 | -5.821 | -3.695 | -10.137 | -9.990 |
| speed_shaping | -54.827 | -54.180 | -53.494 | -52.493 | -51.812 | -53.382 | -59.471 | -51.652 | -65.736 | -51.796 |
| success_bonus | 51.929 | 45.517 | 38.206 | 49.678 | 46.334 | 47.486 | 48.329 | 46.063 | 58.614 | 56.415 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_bonus | 3.4% | 3.4% | 3.6% | 3.6% | 3.5% | 3.4% | 3.7% | 3.4% | 13.2% | 5.6% |
| drift_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |
| fuel_cost | 98.3% | 12.5% | 26.2% | 25.7% | 29.1% | 21.5% | 30.5% | 25.3% | 49.4% | 42.2% |
| landing_quality | 1.7% | 1.5% | 1.5% | 1.8% | 1.7% | 1.5% | 1.9% | 1.5% | 7.4% | 2.9% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_shaping | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| success_bonus | 1.3% | 1.1% | 1.0% | 1.1% | 1.0% | 1.1% | 1.0% | 1.1% | 0.9% | 1.0% |

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
```
