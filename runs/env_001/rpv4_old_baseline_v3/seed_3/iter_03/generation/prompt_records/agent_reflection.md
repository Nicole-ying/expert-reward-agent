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
- current_score: -105.481980
- gap_to_target: 305.481980

# 2. Current reward program (score: -105.481980)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: Progressive landing reward with annealed thresholds + distance shaping + slam prevention
    
    Core insight: contact_bonus was dead because thresholds (y<0.1, x<0.2) were impossibly tight.
    This version uses generous initial thresholds that tighten via training_progress,
    plus intermediate rewards (any contact, proximity) to provide discoverable gradient.
    """
    # --- unpack ---
    x_c, y_c = obs[0], obs[1]
    x_n, y_n = next_obs[0], next_obs[1]
    vx_n, vy_n = next_obs[2], next_obs[3]
    angle_n = next_obs[4]
    angvel_n = next_obs[5]
    lc_n = next_obs[6]
    rc_n = next_obs[7]

    # --- annealed landing thresholds ---
    # Early training: loose thresholds so agent can discover landing reward.
    # Late training: tighten for precision.
    tp = training_progress  # 0.0 → 1.0
    x_thresh = 0.6 - 0.35 * tp   # 0.60 → 0.25
    y_thresh = 0.4 - 0.25 * tp   # 0.40 → 0.15
    vel_thresh = 0.6 - 0.25 * tp # 0.60 → 0.35
    ang_thresh = 0.5 - 0.25 * tp # 0.50 → 0.25

    # --- 1. progress_reward: potential-based delta-distance ---
    dist_c = (x_c ** 2 + y_c ** 2) ** 0.5
    dist_n = (x_n ** 2 + y_n ** 2) ** 0.5
    progress = 3.0 * (dist_c - dist_n)

    # --- 2. proximity_bonus: continuous reward for being near pad ---
    # Provides gradient even before any contact
    proximity = max(0.0, 1.0 - dist_n / 4.0)  # radius 4.0
    proximity_bonus = 0.3 * proximity

    # --- 3. contact_reward: encourage any leg contact ---
    # Intermediate stepping stone toward full landing
    contact_reward = 1.5 * (lc_n + rc_n)

    # --- 4. landing_reward: BIG bonus for full landing state ---
    # Product of soft conditions — only fires when ALL are satisfied
    x_ok = max(0.0, 1.0 - abs(x_n) / x_thresh)
    y_ok = max(0.0, 1.0 - abs(y_n) / y_thresh)
    speed = (vx_n ** 2 + vy_n ** 2) ** 0.5
    vel_ok = max(0.0, 1.0 - speed / vel_thresh)
    ang_ok = max(0.0, 1.0 - abs(angle_n) / ang_thresh)
    both_legs = lc_n * rc_n

    landing_quality = x_ok * y_ok * vel_ok * ang_ok
    landing_reward = 8.0 * landing_quality * both_legs

    # --- 5. orientation_cost: mild regularizer ---
    orient_cost = -0.15 * abs(angle_n) - 0.03 * (angvel_n ** 2)

    # --- 6. slam_cost: penalize high downward speed near ground ---
    near_ground = max(0.0, 1.0 - abs(y_n) / 2.0)
    downward_speed = max(0.0, -vy_n)
    slam_cost = -4.0 * near_ground * downward_speed

    # --- 7. fuel_cost: tiny efficiency incentive ---
    if action == 2:
        fuel_cost = -0.08
    elif action in (1, 3):
        fuel_cost = -0.04
    else:
        fuel_cost = 0.0

    # --- assemble ---
    total_reward = (
        progress
        + proximity_bonus
        + contact_reward
        + landing_reward
        + orient_cost
        + slam_cost
        + fuel_cost
    )

    components = {
        'progress': progress,
        'proximity_bonus': proximity_bonus,
        'contact_reward': contact_reward,
        'landing_reward': landing_reward,
        'orient_cost': orient_cost,
        'slam_cost': slam_cost,
        'fuel_cost': fuel_cost,
    }

    return total_reward, components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-105.481980, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-136.478981, -68.354300]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_bonus | 204.437238 | 73.4% | 73.4% | 100.0% |
| fuel_cost | -47.404000 | -17.0% | 17.0% | 61.0% |
| slam_cost | -20.394881 | -7.3% | 7.3% | 69.9% |
| orient_cost | -5.046639 | -1.8% | 1.8% | 100.0% |
| progress | 0.663680 | 0.2% | 0.5% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_reward | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -532.7 | -489.0 | -153.8 | -134.9 | -128.7 | -125.7 | -113.7 | -104.2 | -107.7 | -106.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| fuel_cost | -7.454 | -45.378 | -47.398 | -47.308 | -45.130 | -45.236 | -47.292 | -47.314 | -47.340 | -47.426 |
| landing_reward | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| orient_cost | -4.737 | -22.167 | -13.022 | -10.629 | -7.213 | -6.172 | -6.020 | -5.504 | -5.647 | -5.284 |
| progress | -5.502 | -7.096 | -0.346 | 0.112 | 0.095 | 0.122 | 0.452 | 0.741 | 0.625 | 0.676 |
| proximity_bonus | 16.336 | 104.451 | 189.880 | 195.771 | 186.824 | 187.493 | 200.862 | 205.031 | 203.907 | 204.557 |
| slam_cost | -1.705 | -1.861 | -6.935 | -11.253 | -11.043 | -11.400 | -16.116 | -21.760 | -19.441 | -20.620 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| fuel_cost | 78.6% | 62.0% | 60.8% | 60.7% | 60.5% | 60.7% | 60.7% | 60.8% | 60.9% | 61.0% |
| landing_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| orient_cost | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_bonus | 95.7% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| slam_cost | 5.7% | 0.9% | 43.2% | 55.8% | 57.1% | 57.8% | 64.6% | 72.0% | 68.8% | 70.1% |

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
