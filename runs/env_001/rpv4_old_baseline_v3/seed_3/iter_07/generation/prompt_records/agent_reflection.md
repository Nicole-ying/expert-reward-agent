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
- current_score: -246.827689
- gap_to_target: 446.827689

# 2. Current reward program (score: -246.827689)
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    xv, yv = obs[2], obs[3]
    angle = obs[4]
    lc, rc = obs[6], obs[7]

    nx, ny = next_obs[0], next_obs[1]
    nxv, nyv = next_obs[2], next_obs[3]
    nangle = next_obs[4]
    nlc, nrc = next_obs[6], next_obs[7]

    dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5

    # ======= A. Progress: delta-distance toward target =======
    w_progress = 8.0
    progress_reward = w_progress * (dist - next_dist)

    # ======= B. Velocity penalty: proximity-gated speed cost =======
    # Active within 2.0m of target, ramps up as agent gets closer
    vel_proximity = max(0.0, 1.0 - next_dist / 2.0)
    next_speed = (nxv**2 + nyv**2) ** 0.5
    w_vel = 2.0
    velocity_penalty = -w_vel * vel_proximity * next_speed

    # ======= C. Stable contact: per-step reward for sustained platform contact =======
    # Only rewards double-contact when near the target (prevents fake landings far away)
    on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    w_contact = 5.0
    stable_contact_reward = w_contact if on_platform else 0.0

    # ======= D. Touchdown transition: small one-time signal for achieving contact =======
    prev_on_platform = (lc > 0.5 and rc > 0.5 and dist < 0.5)
    next_on_platform = (nlc > 0.5 and nrc > 0.5 and next_dist < 0.5)
    touchdown_bonus = 0.0
    if not prev_on_platform and next_on_platform:
        soft_landing = abs(nxv) < 0.5 and abs(nyv) < 0.4 and abs(nangle) < 0.3
        touchdown_bonus = 8.0 if soft_landing else 3.0

    # ======= E. Orientation penalty: discourage tilt =======
    w_orient = 0.3
    orientation_penalty = -w_orient * abs(nangle)

    # ======= F. Fuel efficiency: small cost for engine use =======
    fuel_penalty = -0.05 if action != 0 else 0.0

    total_reward = (
        progress_reward
        + velocity_penalty
        + stable_contact_reward
        + touchdown_bonus
        + orientation_penalty
        + fuel_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "stable_contact_reward": stable_contact_reward,
        "touchdown_bonus": touchdown_bonus,
        "orientation_penalty": orientation_penalty,
        "fuel_penalty": fuel_penalty,
    }
    return float(total_reward), components
```

# 3. Training feedback
# Training Feedback

## Final-policy outcome
score=-246.827689, len=225.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-354.009085, -192.289762]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| velocity_penalty | -17.276114 | -36.0% | 36.0% | 32.7% |
| progress_reward | -5.031623 | -10.5% | 35.2% | 100.0% |
| orientation_penalty | -7.335308 | -15.3% | 15.3% | 100.0% |
| fuel_penalty | -6.502500 | -13.5% | 13.5% | 57.6% |
| stable_contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 5/20
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
| -758.2 | -776.7 | -725.5 | -308.0 | -252.1 | -252.3 | -248.7 | -261.6 | -247.7 | -249.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| fuel_penalty | -5.612 | -6.047 | -5.827 | -8.778 | -10.625 | -12.523 | -7.648 | -6.250 | -6.567 | -6.195 |
| orientation_penalty | -8.757 | -13.740 | -14.781 | -8.906 | -6.600 | -6.883 | -8.676 | -8.346 | -7.602 | -7.443 |
| progress_reward | -25.640 | -22.350 | -17.929 | -9.717 | -6.177 | -5.937 | -5.278 | -5.379 | -4.926 | -5.234 |
| stable_contact_reward | 0.000 | 0.250 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| touchdown_bonus | 0.000 | 0.400 | 0.000 | 0.000 | 0.400 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| velocity_penalty | -16.027 | -18.398 | -16.787 | -17.342 | -18.459 | -16.476 | -16.721 | -17.614 | -16.881 | -16.786 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| fuel_penalty | 100.0% | 98.1% | 100.0% | 100.0% | 99.0% | 100.0% | 59.0% | 57.1% | 57.9% | 57.4% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| stable_contact_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| touchdown_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| velocity_penalty | 46.7% | 46.3% | 53.9% | 41.0% | 28.7% | 25.2% | 26.9% | 32.7% | 30.9% | 31.2% |

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
