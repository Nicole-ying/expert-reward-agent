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
- current_score: -119.624725
- gap_to_target: 319.624725

# 2. Current reward program (score: -119.624725)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nlc, nrc = next_obs[6], next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # Platform detection
    prev_on_platform = (obs[6] > 0.5 and obs[7] > 0.5 and dist < 0.5)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    next_speed = (nxv**2 + nyv**2) ** 0.5

    # === A. Potential-based proximity: difference form prevents accumulation ===
    # Total contribution over full episode bounded to ~8 * (1 - 1/(1+init_dist))
    # Gives zero when hovering at constant distance — no more hovering exploit.
    proximity_reward = 8.0 * (1.0 / (1.0 + next_dist) - 1.0 / (1.0 + dist))

    # === B. Progress: linear distance reduction, supplementary approach signal ===
    progress_reward = 2.0 * (dist - next_dist)

    # === C. Touchdown: large one-time bonus for first platform contact ===
    touchdown_bonus = 50.0 if (not prev_on_platform and on_platform) else 0.0

    # === D. Sustained contact: strong per-step incentive to stay on platform ===
    contact_reward = 20.0 if on_platform else 0.0

    # === E. Speed penalty on platform: discourage sliding after landing ===
    speed_penalty = -5.0 * next_speed if on_platform else 0.0

    # === F. Orientation penalty: mild tilt discouragement ===
    orientation_penalty = -0.2 * abs(nangle)

    # === G. Fuel cost: light regularization for engine use ===
    fuel_cost = -0.02 if action != 0 else 0.0

    total = (proximity_reward + progress_reward + touchdown_bonus +
             contact_reward + speed_penalty + orientation_penalty + fuel_cost)

    components = {
        "proximity_reward": proximity_reward,
        "progress_reward": progress_reward,
        "touchdown_bonus": touchdown_bonus,
        "contact_reward": contact_reward,
        "speed_penalty": speed_penalty,
        "orientation_penalty": orientation_penalty,
        "fuel_cost": fuel_cost,
    }
    return float(total), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-119.624725, len=68.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-151.335058, -24.178362]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| touchdown_bonus | 32.500000 | 54.1% | 54.1% | 0.9% |
| contact_reward | 17.000000 | 28.3% | 28.3% | 1.2% |
| speed_penalty | -3.649612 | -6.1% | 6.1% | 1.2% |
| proximity_reward | 2.941518 | 4.9% | 5.1% | 100.0% |
| progress_reward | 2.230130 | 3.7% | 3.8% | 100.0% |
| orientation_penalty | -1.527330 | -2.5% | 2.5% | 100.0% |
| fuel_cost | -0.047000 | -0.1% | 0.1% | 3.4% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 19/20
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
| -106.3 | -123.1 | -120.9 | -118.9 | -118.5 | -123.7 | -123.8 | -122.6 | -121.1 | -119.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 20.000 | 16.000 | 24.000 | 19.000 | 21.000 | 20.000 | 18.000 | 21.000 | 19.000 | 16.000 |
| fuel_cost | -0.127 | -0.073 | -0.184 | -0.054 | -0.078 | -0.073 | -0.066 | -0.079 | -0.035 | -0.046 |
| orientation_penalty | -2.530 | -1.442 | -1.739 | -0.800 | -0.898 | -1.406 | -1.161 | -1.270 | -1.572 | -1.528 |
| progress_reward | 2.262 | 2.229 | 2.247 | 2.235 | 2.238 | 2.231 | 2.233 | 2.238 | 2.235 | 2.229 |
| proximity_reward | 3.004 | 2.940 | 2.981 | 2.952 | 2.959 | 2.943 | 2.947 | 2.958 | 2.952 | 2.939 |
| speed_penalty | -4.384 | -3.531 | -5.742 | -4.308 | -4.694 | -4.354 | -3.901 | -4.918 | -4.405 | -3.393 |
| touchdown_bonus | 35.000 | 30.000 | 40.000 | 35.000 | 40.000 | 37.500 | 35.000 | 35.000 | 30.000 | 30.000 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 1.4% | 1.2% | 1.8% | 1.4% | 1.5% | 1.5% | 1.3% | 1.5% | 1.4% | 1.2% |
| fuel_cost | 8.9% | 5.3% | 13.4% | 4.0% | 5.7% | 5.3% | 4.8% | 5.8% | 2.6% | 3.4% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 1.4% | 1.2% | 1.8% | 1.4% | 1.5% | 1.5% | 1.3% | 1.5% | 1.4% | 1.2% |
| touchdown_bonus | 1.0% | 0.9% | 1.2% | 1.0% | 1.2% | 1.1% | 1.0% | 1.0% | 0.9% | 0.9% |

# 4. Environment facts
## 1. 任务目标
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: 推测为 float32  
- obs[0]: x_position，水平坐标（相对于目标平台的 x 偏移），reward_usable: true  
- obs[1]: y_position，垂直坐标（相对于平台高度，即飞行器底部到平台表面的距离），reward_usable: true  
- obs[2]: x_velocity，水平线速度，reward_usable: true  
- obs[3]: y_velocity，垂直线速度，reward_usable: true  
- obs[4]: body_angle，身体倾斜角度（弧度 or 归一化角度），reward_usable: true  
- obs[5]: angular_velocity，角速度，reward_usable: true  
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- obs[7]: right_support_contact，右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true  

备注：所有维度均为连续数字，接触标志虽为 0/1 但以浮点形式给出。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine（不点火，靠惯性/重力运动）  
- action 1: left_orientation_engine（点火左向姿态调节引擎，产生向左旋转力矩）  
- action 2: main_engine（点火主引擎，产生向上的推力，通常用于减速/悬停）  
- action 3: right_orientation_engine（点火右向姿态调节引擎，产生向右旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 可能代表身体已静止，结合 y_position≈0 且左右支撑接触为 1 时可视为成功着陆。但 termination 本身是模糊的，因为 crash 后也可能导致静止。  
- failure-like termination: crash_or_body_contact 中“crash”部分，horizontal_position_outside_viewport（水平超出视口）  
- ambiguous termination: body_not_awake_or_settled 的具体触发条件未知，可能是身体进入休眠/静止状态；crash_or_body_contact 中的 body_contact 可能包含正常着陆接触，需依赖位置判断。  
- truncation: 未提供  

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无 success 字段）  
- explicit_failure_flag_available: false（同样无 failure 字段）  
- allowed_info_fields: 无（info 返回空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 internal 字段均不可用，如 termination_reason 等

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action 值（0~3），可用于燃料惩罚或推力鼓励
- other: 通过 next_obs 计算上述量的变化率（如速度变化、位置变化）
```
