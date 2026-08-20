# Duplicate reward retry
The previous generation duplicated iter 2 (runs/env_001/rpv4_old_v4_baseline/seed_4/iter_02/generation/reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v8。

    v7→v8 修改（L2 结构性改动 — 单一语义：恢复 state-reward + baseline 约束框架）：

    问题诊断（v7）：
    v7 的 potential-based goal_proximity + 零基线 settling/attitude 在两个维度上失败：
    1. 导航梯度：potential(d)=5.0/(0.2+d) 的 per-step 梯度在 d>1 时几乎为零，
       agent 无法获得有效的方向引导。
    2. 零基线约束：settling 和 attitude 的 multiplier 完全由 prox_gate 决定，
       远处 penalty≈0，agent 高速飞行 → 68 步 crash。

    证据：v7 所有 checkpoint 完全平坦（score -118→-122，零学习），
    20/20 early terminal，landing_contact 活跃率仅 1.0%。

    v8 方案：恢复到 v2 的已验证框架（v2 score=178.70，距目标仅 ~10%）：
    1. goal_proximity: 恢复 `3.0 * dist_improvement + 1.0 * exp(-dist/0.5)`
       - delta 项提供均匀的导航梯度（不受距离衰减）
       - proximity_bonus 在目标附近提供额外引力
       - delta 项总量 bounded（≈initial_dist - final_dist），天然防悬停
    2. settling_stability: 恢复基线 `0.2 + 2.5*prox_gate`，确保远处也有最低速度约束
    3. attitude_constraint: 恢复基线 `0.3 + 2.0*prox_gate`，确保远处也有最低姿态约束
    4. landing_contact: 保持 2.0/腿（v2 验证有效，dual-leg = 4.0/步）
    5. fuel_efficiency: 保持 -0.05

    v2 的 proximity_bonus=1.0 理论上存在悬停风险，但 v2 实际运行中 landing=4.0/步
    远超 proximity=1.0/步，且 settling 持续施加速度惩罚，悬停未实际发生（404 步正常终止）。
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

    # ========== Component A: goal_proximity ==========
    # 改进量：每步距离变化，总量 bounded ≈ initial_dist - final_dist
    dist_improvement = current_dist - next_dist

    # 指数邻近奖励：在目标附近提供每步强梯度
    # exp(-dist/0.5): dist=0→1.0, dist=0.5→0.37, dist=1.0→0.14, dist=2.0→0.02
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.2→0.5, dist=1.0→0.17, dist=2.0→0.09
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability ==========
    # 基线 0.2 确保远处也有最低速度惩罚，prox_gate 在近处增强
    # 远处: multiplier≈0.2+0.23=0.43, 近处: multiplier≈0.2+2.5=2.7
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 基线 0.3 确保远处也有最低姿态约束
    # 远处: multiplier≈0.3+0.18=0.48, 近处: multiplier≈0.3+2.0=2.3
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact ==========
    # 目标附近 + 低速时，双足着陆 = 4.0/步，远超 proximity_bonus 的 1.0/步
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

下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (score=-122.805639)
# Training Feedback

## Final-policy outcome
score=-122.805639, len=68.300000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-146.002977, -100.809739]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -20.666787 | -61.3% | 61.3% | 100.0% |
| goal_proximity | 8.021252 | 23.8% | 25.5% | 100.0% |
| landing_contact | 2.400000 | 7.1% | 7.1% | 1.0% |
| attitude_constraint | -1.923012 | -5.7% | 5.7% | 100.0% |
| fuel_efficiency | -0.142500 | -0.4% | 0.4% | 4.2% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| -117.9 | -119.4 | -118.5 | -121.0 | -123.9 | -120.2 | -124.5 | -120.1 | -122.8 | -122.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -3.721 | -3.035 | -2.823 | -2.841 | -2.547 | -1.498 | -2.006 | -1.549 | -1.804 | -1.849 |
| fuel_efficiency | -0.185 | -0.263 | -0.217 | -0.150 | -0.130 | -0.138 | -0.168 | -0.165 | -0.155 | -0.147 |
| goal_proximity | 8.006 | 8.018 | 8.027 | 8.012 | 7.974 | 8.040 | 8.017 | 8.054 | 8.030 | 8.025 |
| landing_contact | 4.700 | 3.600 | 3.700 | 3.500 | 2.500 | 2.400 | 2.100 | 2.400 | 2.700 | 2.400 |
| settling_stability | -20.806 | -20.779 | -20.757 | -20.775 | -20.716 | -20.688 | -20.656 | -20.645 | -20.648 | -20.647 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 5.3% | 7.6% | 6.3% | 4.4% | 3.8% | 4.0% | 4.9% | 4.8% | 4.5% | 4.3% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 2.7% | 1.8% | 1.8% | 1.7% | 1.1% | 1.0% | 0.9% | 1.0% | 1.1% | 1.0% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v7。

    v6→v7 修改（L2 结构性改动 — 单一语义：消除 per-step state reward 的悬停 exploit，
    同时恢复导航梯度）：

    问题诊断（v6）：
    v6 的 goal_proximity 依赖 `0.3 * exp(-dist/0.5)` 作为 per-step state reward，
    指数衰减使 dist>1.0 时梯度几乎为零。agent 不点火（fuel 4.1%）→ crash（68步）。
    所有 checkpoint 完全平坦——zero learning。

    同时 settling_stability 的 `0.2 + 2.5*prox_gate` baseline 在远处也对速度施加惩罚，
    进一步抑制导航动机。

    更深层根因：v2-v6 共享的 `delta + coeff*exp(-dist/tau)` 框架存在内在矛盾——
    per-step state reward 总量与 episode 长度正相关，必然产生悬停 exploit。
    v2 侥幸成功仅因 landing=4.0 暂时压制了悬停动机，v5 暴露了脆弱性。

    v7 方案（三项 L2 结构性改动）：
    1. goal_proximity: 从 state reward 改为 potential-based shaping。
       potential(d) = 5.0/(0.2+d)，reward = Φ(next) - Φ(current)。
       每步提供梯度，episode 总量 bounded（≈22 from dist=2→0），悬停零积累。
    2. settling_stability: 去除 0.2 baseline。multiplier = 3.0*prox_gate。
       远处≈无惩罚，近处强约束。让 agent 先自由导航，再近处减速。
    3. attitude_constraint: 去除 0.3 baseline。multiplier = 2.5*prox_gate。
       同理——远处不约束角度，近处严格要求姿态。

    landing_contact 保持 2.0/腿（v2 验证有效），fuel 保持 -0.05。
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

    # ========== Component A: goal_proximity (v7: potential-based) ==========
    # v7: 用 potential shaping 替代 state reward。
    # potential(d) = 5.0 / (0.2 + d):
    #   d=0   → 25.0
    #   d=0.5 →  7.14
    #   d=1.0 →  4.17
    #   d=2.0 →  2.27
    #   d=5.0 →  0.96
    # 奖励 = Φ(next) - Φ(current)，靠近目标为正，远离为负。
    # Episode 总量 bounded ≈ potential(0) - potential(initial)，与步数无关。
    # 悬停时 Φ(next)=Φ(current) → 奖励为 0，无 exploit 可能。
    def potential(dist):
        return 5.0 / (0.2 + dist)

    goal_proximity = potential(next_dist) - potential(current_dist)

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.2→0.5, dist=1.0→0.17, dist=2.0→0.09
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability (v7: 零基线) ==========
    # v7: 去除 0.2 baseline。multiplier 从 0.2+2.5*prox_gate → 3.0*prox_gate。
    # 远处（dist>2, prox_gate≈0.09）：penalty ≈ 0.11/speed_unit，几乎不惩罚。
    # 近处（dist=0, prox_gate=1.0）：penalty ≈ 1.2/speed_unit，强减速。
    settling_multiplier = 3.0 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint (v7: 零基线) ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # v7: 去除 0.3 baseline。multiplier 从 0.3+2.0*prox_gate → 2.5*prox_gate。
    # 远处几乎不惩罚角度，近处严格要求姿态平稳。
    attitude_multiplier = 2.5 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact ==========
    # 保持 2.0/腿（v2 验证有效）。双足着陆 = 4.0/步。
    # 由于 goal_proximity 是 potential-based（悬停=0），不存在悬停收割竞争。
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
