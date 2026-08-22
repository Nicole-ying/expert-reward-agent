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
- current_score: -112.760626
- gap_to_target: 312.760626

# 2. Current reward program (score: -112.760626)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Return to iteration-2 architecture (score +40, 850-step survival) with
    enhancements: terminal success bonus, boosted contact/landing rewards.

    Core principle: positive shaping drives descent; penalties are kept
    gentle so the agent survives long enough to discover landing.
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
    # 1. Potential-based approach — primary descent driver.
    #    Bounded total = w_app * initial_distance.  Unexploitable.
    #    Weight 5.0 (same as iteration 2, proven safe).
    # ------------------------------------------------------------------
    w_app = 5.0
    approach_reward = w_app * (dist_prev - dist)

    # ------------------------------------------------------------------
    # 2. Small proximity reward — continuous positive signal near pad.
    #    w_prox / (1+dist):  at dist=0 → 0.05,  at dist=5 → 0.008.
    #    Same formula as iteration 2 where it contributed +33 over 850 steps.
    #    NOT the exponential form from iteration 3 that caused hovering.
    # ------------------------------------------------------------------
    w_prox = 0.05
    proximity = w_prox / (1.0 + dist)

    # ------------------------------------------------------------------
    # 3. Gentle universal speed cost — discourages reckless speed.
    #    Same as iteration 2: -0.1 * speed.  At speed=3.0 this is -0.3/step.
    #    Survived 850 steps with this in iteration 2.
    # ------------------------------------------------------------------
    w_vel = 0.1
    vel_cost = -w_vel * speed

    # ------------------------------------------------------------------
    # 4. Quadratic orientation cost — keep lander upright.
    #    angle**2 is gentler than abs(angle) for small tilts, still
    #    punishes large deviations.  Same as iteration 2.
    # ------------------------------------------------------------------
    w_angle = 0.5
    w_angvel = 0.1
    orientation_cost = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ------------------------------------------------------------------
    # 5. Per-foot contact bonus — encourage ground contact.
    #    Boosted to 1.0/foot (from 0.5 in iteration 2).
    # ------------------------------------------------------------------
    w_contact = 1.0
    contact_bonus = w_contact * (left + right)

    # ------------------------------------------------------------------
    # 6. Landing quality — continuous bonus while both feet are down
    #    and the lander is stable.  Boosted to 5.0 (from 2.0 in iter 2).
    #    Perfect landing (angle=0, speed=0): 5.0/step.
    # ------------------------------------------------------------------
    both = left * right
    w_land = 5.0
    landing_quality = w_land * both / (1.0 + abs(angle) + speed)

    # ------------------------------------------------------------------
    # 7. Terminal success bonus — one-shot, quality-scaled.
    #    Fires on the transition to both-feet-down, near the pad.
    #    Wider radius (1.5) for easier discovery; continuous quality
    #    scaling so imperfect landings get partial credit.
    # ------------------------------------------------------------------
    prev_both = obs[6] * obs[7]
    now_both = left * right
    just_landed = float(now_both > 0.5 and prev_both < 0.5)
    near_pad = float(dist < 1.5)
    land_quality = 1.0 / (1.0 + abs(angle) + speed)
    w_success = 200.0
    success_bonus = w_success * just_landed * near_pad * land_quality

    # ------------------------------------------------------------------
    # 8. Fuel efficiency — light pressure against unnecessary engine use.
    #    Same as iteration 2.
    # ------------------------------------------------------------------
    w_fuel = 0.03
    fuel_cost = -w_fuel * float(action != 0)

    total_reward = (
        approach_reward + proximity + vel_cost + orientation_cost +
        contact_bonus + landing_quality + success_bonus + fuel_cost
    )

    components = {
        "approach_reward": approach_reward,
        "proximity": proximity,
        "vel_cost": vel_cost,
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
score=-112.760626, len=69.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-212.671386, 17.552794]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| success_bonus | 76.199523 | 74.5% | 74.5% | 1.0% |
| vel_cost | -7.383710 | -7.2% | 7.2% | 100.0% |
| approach_reward | 5.596897 | 5.5% | 5.7% | 100.0% |
| contact_bonus | 4.300000 | 4.2% | 4.2% | 4.8% |
| orientation_cost | -3.679135 | -3.6% | 3.6% | 100.0% |
| landing_quality | 2.876069 | 2.8% | 2.8% | 1.4% |
| proximity | 1.864612 | 1.8% | 1.8% | 100.0% |
| fuel_cost | -0.145500 | -0.1% | 0.1% | 6.9% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 18/20
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
| -111.6 | -118.5 | -120.2 | -114.0 | -107.1 | -112.2 | -118.1 | -108.4 | -115.4 | -111.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 5.593 | 5.586 | 5.575 | 5.598 | 5.577 | 5.610 | 5.599 | 5.601 | 5.605 | 5.596 |
| contact_bonus | 3.300 | 3.100 | 3.300 | 3.200 | 3.300 | 3.250 | 3.150 | 3.500 | 3.250 | 4.300 |
| fuel_cost | -0.198 | -0.102 | -0.168 | -0.417 | -0.204 | -0.477 | -0.182 | -0.179 | -0.234 | -0.113 |
| landing_quality | 3.264 | 3.296 | 2.761 | 3.022 | 3.483 | 3.346 | 3.272 | 4.041 | 3.386 | 2.717 |
| orientation_cost | -1.496 | -2.374 | -2.840 | -1.731 | -2.159 | -1.394 | -2.474 | -1.239 | -1.941 | -3.858 |
| proximity | 1.820 | 1.815 | 1.816 | 1.819 | 1.822 | 1.819 | 1.816 | 1.823 | 1.822 | 1.864 |
| success_bonus | 74.204 | 67.088 | 68.617 | 70.056 | 68.706 | 73.708 | 70.126 | 82.780 | 84.263 | 72.240 |
| vel_cost | -7.323 | -7.329 | -7.350 | -7.309 | -7.335 | -7.298 | -7.310 | -7.304 | -7.300 | -7.390 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_bonus | 3.2% | 2.9% | 3.4% | 3.1% | 3.1% | 3.1% | 2.9% | 3.1% | 3.1% | 4.8% |
| fuel_cost | 9.6% | 5.0% | 8.2% | 20.3% | 9.9% | 23.2% | 8.9% | 8.7% | 11.4% | 5.4% |
| landing_quality | 1.6% | 1.6% | 1.5% | 1.6% | 1.8% | 1.7% | 1.7% | 2.0% | 1.7% | 1.4% |
| orientation_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| success_bonus | 1.1% | 1.1% | 1.1% | 1.2% | 1.1% | 1.2% | 1.2% | 1.3% | 1.2% | 1.0% |
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
