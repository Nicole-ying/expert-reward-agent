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
- current_score: -10.305339
- gap_to_target: 210.305339

# 2. Current reward program (score: -10.305339)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Iteration-2 core architecture (proven: score +40→+88, 850-step survival)
    + main-engine deceleration bonus (direct incentive to slow descent)
    + crash penalty (detect hard ground contact)
    - NO success_bonus (lottery destabilizer removed)
    """
    x_prev, y_prev = obs[0], obs[1]
    x, y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left = next_obs[6]
    right = next_obs[7]

    vy_prev = obs[3]
    prev_left = obs[6]
    prev_right = obs[7]

    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist = (x ** 2 + y ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5

    # ------------------------------------------------------------------
    # 1. Potential-based approach — ITERATION 2 WEIGHT (proven safe)
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Proximity — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Velocity cost — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 4. Orientation cost — ITERATION 2 WEIGHTS
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 5. Per-foot contact bonus — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_contact = 0.5
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 6. Landing quality — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    both_feet = left * right
    w_land = 2.0
    landing_quality = w_land * both_feet / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 7. Fuel efficiency — ITERATION 2 WEIGHT
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    # ------------------------------------------------------------------
    # 8. Main-engine deceleration bonus (NEW — directly rewards slowing descent)
    #    vy - vy_prev > 0  means vertical speed became less negative (decelerating).
    #    Only rewards when action==2 (main engine).  This gives the agent a clear
    #    gradient to discover that main engine controls descent speed.
    # ------------------------------------------------------------------
    decel = vy - vy_prev
    using_main = float(action == 2)
    w_main_decel = 2.0
    main_decel_bonus = w_main_decel * max(0.0, decel) * using_main

    # ------------------------------------------------------------------
    # 9. Crash penalty (NEW — penalises first ground contact at high speed)
    #    Uses pre-contact speed so the penalty reflects impact velocity,
    #    not post-collision-resolved speed.
    # ------------------------------------------------------------------
    prev_any_contact = float(prev_left + prev_right > 0.5)
    cur_any_contact = float(left + right > 0.5)
    just_made_contact = float(cur_any_contact > 0.5 and prev_any_contact < 0.5)
    impact_speed = max(0.0, prev_speed - 1.0)
    w_crash = 25.0
    crash_penalty = -w_crash * just_made_contact * impact_speed

    total_reward = (
        approach_reward + proximity + vel_cost + orientation_cost +
        contact_bonus + landing_quality + fuel_cost +
        main_decel_bonus + crash_penalty
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "vel_cost": vel_cost,
        "orientation_cost": orientation_cost,
        "contact_bonus": contact_bonus,
        "landing_quality": landing_quality,
        "fuel_cost": fuel_cost,
        "main_decel_bonus": main_decel_bonus,
        "crash_penalty": crash_penalty,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-10.305339, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-39.145883, 27.549397]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity | 37.139062 | 37.9% | 37.9% | 100.0% |
| main_decel_bonus | 22.937084 | 23.4% | 23.4% | 53.1% |
| fuel_cost | -19.137000 | -19.5% | 19.5% | 63.8% |
| vel_cost | -10.869547 | -11.1% | 11.1% | 100.0% |
| approach_reward | 5.907605 | 6.0% | 7.0% | 100.0% |
| orientation_cost | -0.963790 | -1.0% | 1.0% | 100.0% |
| contact_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_quality | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -112.1 | -51.8 | -78.7 | -106.5 | -0.5 | -23.4 | -57.9 | -20.2 | -2.4 | -9.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 2.334 | 2.787 | 3.581 | 2.922 | 5.928 | 5.683 | 3.477 | 4.581 | 5.437 | 6.023 |
| contact_bonus | 13.800 | 16.025 | 10.250 | 0.600 | 13.400 | 3.775 | 15.000 | 19.500 | 10.250 | 0.000 |
| crash_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fuel_cost | -15.383 | -12.918 | -21.522 | -21.230 | -26.081 | -24.998 | -24.537 | -20.045 | -17.379 | -19.357 |
| landing_quality | 14.439 | 15.660 | 13.130 | 0.380 | 17.414 | 4.588 | 14.490 | 29.849 | 12.938 | 0.000 |
| main_decel_bonus | 12.037 | 11.308 | 19.633 | 16.505 | 20.070 | 22.233 | 18.642 | 21.743 | 20.229 | 22.868 |
| orientation_cost | -1.664 | -2.466 | -2.158 | -1.500 | -1.890 | -1.412 | -2.332 | -1.937 | -1.649 | -0.860 |
| proximity | 16.837 | 16.097 | 27.438 | 22.612 | 33.608 | 35.922 | 27.738 | 32.704 | 33.149 | 38.295 |
| vel_cost | -12.412 | -11.269 | -11.285 | -12.295 | -10.692 | -10.590 | -12.971 | -10.535 | -10.220 | -10.774 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% |
| contact_bonus | 3.2% | 4.1% | 1.3% | 0.1% | 1.7% | 0.4% | 2.4% | 2.1% | 1.3% | 0.0% |
| crash_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fuel_cost | 94.5% | 83.6% | 82.5% | 97.5% | 97.0% | 85.2% | 97.7% | 68.7% | 64.6% | 64.5% |
| landing_quality | 1.9% | 2.2% | 1.1% | 0.0% | 1.3% | 0.3% | 1.2% | 2.0% | 1.0% | 0.0% |
| main_decel_bonus | 51.3% | 51.0% | 52.5% | 53.4% | 52.5% | 53.0% | 51.8% | 52.1% | 52.5% | 53.1% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
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
```
