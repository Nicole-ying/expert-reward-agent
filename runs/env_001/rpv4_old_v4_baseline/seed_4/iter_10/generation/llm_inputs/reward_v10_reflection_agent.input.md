下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (score=63.660194)
# Training Feedback

## Final-policy outcome
score=63.660194, len=955.500000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-36.988538, 150.857897]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 897.904747 | 64.4% | 64.4% | 100.0% |
| landing_contact | 399.600000 | 28.6% | 28.6% | 11.9% |
| fuel_efficiency | -46.075000 | -3.3% | 3.3% | 96.4% |
| settling_stability | -36.821348 | -2.6% | 2.6% | 100.0% |
| attitude_constraint | -14.506255 | -1.0% | 1.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)

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
| -765.9 | -939.5 | -301.4 | -180.8 | -27.9 | -70.1 | -2.2 | 62.3 | 38.7 | -18.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -8.303 | -10.284 | -7.018 | -10.183 | -9.266 | -8.804 | -12.557 | -13.898 | -14.114 | -11.137 |
| fuel_efficiency | -8.623 | -14.720 | -21.388 | -30.543 | -39.647 | -49.820 | -49.077 | -43.030 | -44.200 | -50.000 |
| goal_proximity | 62.079 | 115.122 | 364.829 | 596.988 | 860.243 | 811.121 | 936.386 | 825.280 | 835.428 | 926.397 |
| landing_contact | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 77.600 | 289.600 | 294.100 | 0.000 |
| settling_stability | -13.558 | -15.920 | -14.493 | -11.235 | -19.814 | -15.248 | -31.393 | -25.036 | -25.152 | -28.361 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 79.7% | 69.4% | 62.2% | 61.1% | 79.3% | 99.6% | 98.2% | 96.2% | 97.7% | 100.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 2.2% | 9.6% | 9.6% | 0.0% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v9。

    v8→v9 修改（L2 结构性改动 — 单一语义：Lorentzian宽覆盖邻近函数 + 零基线速度约束）：

    问题诊断（v8）：
    v8 的 goal_proximity = 3.0*Δ + 0.3*vel_toward。
    checkpoint 全平坦：goal_proximity 16.60→16.74，settling_stability -17.51→-17.26。
    二者几乎完全抵消（+16.74 vs -17.26），净导航信号 ≈ 0，agent 永远无法学习。

    根因分析：
    1. vel_toward 对随机策略均值为零——agent 永远发现不了"靠近目标=好"
    2. settling 的 0.2 baseline 在远处也对速度施加惩罚，与 goal_proximity 中
       有限的 dist_improvement 信号相互抵消

    v2 成功的关键：exp(-dist/0.5) 是 per-step 位置奖励，随机探索中偶然靠近目标时
    获得更多奖励，形成可学习的梯度盆地。但 v2 的指数衰减太快（dist=2→0.018），
    中等距离处梯度已接近零。

    v9 方案（两项联动，同一语义——恢复导航梯度）：
    1. goal_proximity: 用 Lorentzian/Cauchy 函数 1/(1+(d/τ)²) 替代 exp(-d/τ)。
       τ=2.0 提供宽覆盖：dist=0→1.0, dist=1→0.8, dist=2→0.5, dist=5→0.138。
       相比 exp(-d/0.5) 在 dist=2 处 0.018，Lorentzian 在中等距离提供 28 倍更强的信号。
       函数族完全不同（有理函数 vs 指数函数），且历史上从未使用。
    2. settling_stability: 移除 0.2 baseline，multiplier 从 0.2+2.5*prox_gate
       改为 3.0*prox_gate。远处速度惩罚大幅减弱，不再抵消导航信号。
       近处（prox_gate→1.0）约束不变强。

    landing_contact=2.0/腿 保持 v2 验证值。其余组件不变。
    """

    # ========== 信号提取 ==========
    x_pos, y_pos = obs[0], obs[1]
    x_vel, y_vel = obs[2], obs[3]
    body_angle = obs[4]
    ang_vel = obs[5]
    left_contact, right_contact = obs[6], obs[7]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_ang_vel = next_obs[5]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # ========== 距离与速度 ==========
    current_dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    next_speed = (next_x_vel ** 2 + next_y_vel ** 2) ** 0.5

    # ========== Component A: goal_proximity (v9: Lorentzian proximity) ==========
    # 距离改进量：每步靠近量，总量 bounded ≈ initial_dist - final_dist
    dist_improvement = current_dist - next_dist

    # v9: Lorentzian/Cauchy 邻近奖励 —— 历史上首次使用有理函数族。
    # 1/(1+(d/τ)²) with τ=2.0:
    #   d=0.0 → 1.000    d=0.5 → 0.941    d=1.0 → 0.800
    #   d=2.0 → 0.500    d=3.0 → 0.308    d=5.0 → 0.138
    # 对比 v2 的 exp(-d/0.5): d=2.0→0.018, d=5.0→0.000045
    # Lorentzian 在中等距离提供 28-3000x 更强的信号，形成宽漏斗效应。
    # 函数族：有理/代数衰减 vs 指数衰减 —— 数学上完全不同。
    tau = 2.0
    proximity_bonus = 1.0 / (1.0 + (next_dist / tau) ** 2)

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.5→0.286, dist=1.0→0.167, dist=2.0→0.091
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability (v9: 零基线) ==========
    # v9: 移除 0.2 baseline。multiplier = 3.0*prox_gate（原为 0.2+2.5*prox_gate）。
    # 远处（dist=5, prox_gate≈0.038）：multiplier≈0.115，speed=3→penalty≈-0.10/步
    #   vs v8: multiplier=0.2+0.096=0.296，penalty≈-0.27/步 —— 减弱 63%
    # 近处（dist=0, prox_gate=1.0）：multiplier=3.0，与原 v8 的 2.7 基本持平。
    settling_multiplier = 3.0 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint (保持 v8 框架) ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 基线 0.3 确保远处也有最低姿态约束（量级小，不干扰导航）
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact (v2 验证有效) ==========
    # 目标附近 + 低速时，双足着陆 = 4.0/步
    # 在 dist=0 处：proximity_bonus=1.0 vs landing=4.0，4:1 压制悬停（同 v2 安全比）
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0

    landing_contact = 0.0
    if near_target and low_speed:
        landing_contact = 2.0 * (next_left_contact + next_right_contact)

    # ========== Component E: fuel_efficiency ==========
    fuel_efficiency = -0.05 * float(action != 0)

    # ========== 汇总 ==========
    total_reward = (
        goal_proximity +
        settling_stability +
        attitude_constraint +
        landing_contact +
        fuel_efficiency
    )

    components = {
        'goal_proximity': goal_proximity,
        'settling_stability': settling_stability,
        'attitude_constraint': attitude_constraint,
        'landing_contact': landing_contact,
        'fuel_efficiency': fuel_efficiency
    }

    return float(total_reward), components
```

# 反思规则

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

分析完每个组件后再写代码。

### 输出要求

Please analyze each existing reward component in the suggested manner above first, and then write the reward function code.
