# Duplicate reward retry
The previous generation duplicated iter 2 (runs/env_001/rpv4_old_v4_baseline/seed_4/iter_02/generation/reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v7。

    v6→v7 修改（L1 三系数联动 — 单一语义：恢复 v2 有效梯度结构）：

    问题诊断（v6）：
    v6 将 proximity_bonus 从 1.0 压至 0.3，意图消灭 v5 的悬停陷阱。
    结果导航信号崩溃：goal_proximity ep_sum 从 v2 的 193 暴跌至 7.43（1/26），
    agent 在 68 步内 crash，20/20 early terminal。checkpoint 趋势完全平坦
    （cp10%-cp100%: 7.40→7.43），典型的"无学习梯度"模式（与 v1 同）。

    根因：proximity_bonus=0.3 弱到无法克服 settling_stability 的运动惩罚，
    agent 选择不点火（fuel active 从 82.8%→4.1%）→ 自由落体 → crash。

    v2 整套参数已验证有效：score=178.7, len=404, 20/20 terminated。
    v6 偏离了这条成功路径，只需回归。

    v7 方案（L1 三系数联动，回归 v2）：
    1. proximity_bonus: 0.3 → 1.0（恢复导航"漏斗"梯度）
    2. settling_stability 速度系数: -0.4 → -0.3（v2 原值）
    3. attitude_constraint 角度系数: 0.4 → 0.3（v2 原值）
    
    landing_contact 保持 2.0/leg（4:1 着陆 vs 悬停梯度比，v2 验证有效）。
    其余不变。

    预期：恢复 v2 行为 — agent 主动导航→着陆→终止，score → 175-195。
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
    # 改进量（保持 credit assignment 一致性）
    dist_improvement = current_dist - next_dist

    # v7: proximity_bonus 回归 1.0（v2 原值）
    # exp(-dist/0.5): dist=0→1.0, dist=0.5→0.37, dist=1.0→0.14, dist=2.0→0.02
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.2→0.5, dist=1.0→0.17
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability ==========
    # v7: 速度系数回归 -0.3（v2 原值），邻近时增强至 ~2.7x
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # v7: 角度系数回归 0.3（v2 原值）
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact ==========
    # v7: 保持 2.0/腿（v2/v6 验证）
    # 双足着陆 = 4.0/步 vs proximity_bonus 1.0/步 = 4:1 梯度比
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

# 上一轮训练反馈 (score=-115.554859)
# Training Feedback

## Final-policy outcome
score=-115.554859, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-138.753007, -96.731262]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -23.002279 | -65.9% | 65.9% | 100.0% |
| goal_proximity | 7.431531 | 21.3% | 21.3% | 100.0% |
| landing_contact | 3.100000 | 8.9% | 8.9% | 1.3% |
| attitude_constraint | -1.208334 | -3.5% | 3.5% | 100.0% |
| fuel_efficiency | -0.140000 | -0.4% | 0.4% | 4.1% |

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
| -124.3 | -119.6 | -123.3 | -124.0 | -122.9 | -117.0 | -113.6 | -114.8 | -115.4 | -114.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -5.651 | -3.445 | -2.734 | -2.346 | -2.293 | -1.439 | -1.113 | -1.270 | -1.266 | -1.192 |
| fuel_efficiency | -0.215 | -0.147 | -0.098 | -0.105 | -0.128 | -0.128 | -0.145 | -0.138 | -0.140 | -0.138 |
| goal_proximity | 7.625 | 7.438 | 7.405 | 7.404 | 7.406 | 7.439 | 7.452 | 7.442 | 7.442 | 7.434 |
| landing_contact | 4.000 | 3.000 | 2.400 | 2.300 | 2.100 | 2.400 | 2.900 | 2.500 | 2.800 | 3.200 |
| settling_stability | -23.436 | -23.186 | -23.113 | -23.104 | -23.104 | -23.148 | -23.119 | -23.139 | -23.106 | -22.988 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 6.1% | 4.3% | 2.9% | 3.1% | 3.7% | 3.7% | 4.2% | 4.0% | 4.1% | 4.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 2.2% | 1.5% | 1.0% | 1.0% | 0.9% | 1.0% | 1.2% | 1.0% | 1.2% | 1.3% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v6。

    v5→v6 修改（L1 双系数联动 — 单一语义：消灭悬停陷阱）：

    问题诊断（v5）：
    v5 将 landing 从 0.5→1.0/腿，意图恢复「着陆 > 盘旋」梯度（2.0 vs 1.0）。
    结果 score 从 172.9 暴跌至 -5.9，20/20 截断，landing 完全死亡。

    根因不是 landing 系数太小，而是 proximity_bonus = 1.0 * exp(-dist/0.5)
    作为无上限的 per-step 状态奖励，悬停 1000 步可收割 ~370-1000，
    远超 landing 可获取总量（即使双足着陆 4.0/步 × ~50 步 = 200）。
    agent 在 cp60% 学会着陆后（landing=94），cp80%-100% 发现了更优的悬停策略，
    于是完全放弃着陆 → landing 归零，所有 episode 跑满 1000 步截断。

    证据链：
    - goal_proximity ep_sum 与 episode 长度完全正相关：v2(404步)=193 → v4(663)=391 → v5(1000)=643
    - v5 checkpoint: cp60% landing=94.1 → cp80%=48.4 → cp100%=0（典型的策略替换）
    - v2（landing=2.0, proximity=1.0, 4:1 比）在 178.7 正常工作

    v6 方案（L1 双系数联动）：
    1. proximity_bonus: 1.0 → 0.3（削弱悬停的每步收益）
    2. landing_contact: 1.0/腿 → 2.0/腿（恢复 v2 验证有效的着陆激励）
    
    着陆 max 4.0/步 vs 悬停 proximity 0.3/步 = 13:1 梯度比，
    远超 v2 的 4:1，安全边际充足。其余组件不变。

    预期：landing 恢复活跃，agent 主动着陆终结 episode，len 回落至 400-500，
    score → 185-195。
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
    # 改进量（保持 credit assignment 一致性，不受悬停影响）
    dist_improvement = current_dist - next_dist

    # v6: 系数从 1.0 → 0.3，削弱悬停的每步收益
    # exp(-dist/0.5): dist=0→0.3, dist=0.5→0.11, dist=1.0→0.04
    # 悬停 1000 步最多收割 300（vs v5 的 1000），不再能淹没 landing 信号
    proximity_bonus = 0.3 * (2.718281828 ** (-next_dist / 0.5))

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability ==========
    # 保持不变：速度惩罚，邻近目标时增强
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.4 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 保持不变：角度 + 角速度惩罚，邻近目标时增强
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.4 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact (v6: 回归 2.0/腿) ==========
    # v6: 系数从 1.0/腿 → 2.0/腿。回归 v2 验证有效的设置。
    # 双足着陆 = 4.0/步 vs proximity_bonus 0.3/步 = 13:1 梯度比。
    # agent 有极强动机完成着陆并终结 episode，而非悬停收割 proximity。
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
