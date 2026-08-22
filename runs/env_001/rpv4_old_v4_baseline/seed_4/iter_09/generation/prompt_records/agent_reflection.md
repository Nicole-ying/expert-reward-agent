# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的数据和原则，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action
- `read_training_feedback(N)` — 某轮训练的详细反馈：分数、每个组件的 ep_sum/active_rate/magnitude_share
- `read_reward_code(N)` — 某轮的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某组件在所有迭代中的 ep_sum_mean 和 active_rate 演化
- `read_environment_card()` — 环境信息：任务目标、观察空间（8维）、动作空间（4动作）、终止条件
- `read_checkpoint_trend(N)` — 某轮训练中各组件在不同训练阶段的趋势

# 修改层级

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态。常见模式参考：

| 证据模式 | 参考变换 | 验证标准 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | 稀疏→连续过程证据 | active_rate 改善，不产生 proxy 徘徊 |
| 极端值支配奖励 | 无界→归一化有界 | 极端轨迹支配下降，方差下降 |
| 占据好状态即可持续获奖 | 状态值→改善量/势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | 全局→阶段门控 | 早期探索与局部约束同时改善 |
| 多个小因子相乘导致塌缩 | 乘积→加性/几何平均 | 非零反馈增多，联合约束保留 |
| 持续事件被重复领取 | 持续→状态转移事件 | 重复积累下降，外部完成保持 |
| proxy 提高但外部任务不升 | 代理→任务完成对齐 | proxy 与外部分数重新同向 |
| 稠密 proxy 形成中分平台 | 全程 proxy→局部/转移任务信号 | 刷新 best，完成相关行为增加 |

- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。

# 关键规则

- **避免打转**：如果 `read_past_reflection` 显示之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先 L1**：如果当前得分或 best 得分接近目标（差距 < 10%），优先 L1 微调。因为之前的经验发现已多次出现分数接近目标时做 L2，结果越改越差。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

```

## User Prompt

```markdown
# Duplicate reward retry
The previous generation duplicated iter 2 (runs/env_001/rpv4_old_v4_baseline/seed_4/iter_02/generation/reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v9。

    v8→v9 修改（L2 结构性改动 — 单一语义：恢复 exp(-dist/tau) 状态值导航框架）：

    问题诊断（v6-v8 全系列）：
    v6: proximity_bonus 0.3*exp(-dist/0.5) → 梯度太弱，goal_proximity=7.4，crash
    v7: potential-based shaping 5/(0.2+d) → 远处梯度为零，goal_proximity=8.0，crash
    v8: vel_toward 方向投影 → 纯"改善量"信号，随机探索时均值为零，
         agent 永远发现不了"靠近目标=好"这个基本事实，goal_proximity=16.7，crash

    根因：v6-v8 的所有替代方案都缺少「状态值」信号——
    一个只需"处于某位置"即可获得的正向反馈，不依赖主动朝目标移动。
    v2 的 exp(-dist/0.5) 提供了这个信号：随机探索中偶然靠近目标时获得奖励，
    形成可学习的梯度盆地。这是唯一在历史上成功导航的框架（v2: 178.7）。

    v5 的悬停 exploit 不是 v2 框架的错——是因为 landing 被削弱到 1.0/腿（ratio 2:1）。
    v2 的 landing=2.0/腿（ratio 4:1）是安全的，agent 偏好着陆而非悬停。

    v9 方案：goal_proximity 从 vel_toward 回退到 exp(-dist/0.5)（v2 原版），
    其余组件不变。预期 goal_proximity ep_sum 从 16.7 恢复到 ~190，
    settling 占比从 44% 回落到 ~5%，landing 恢复活跃，score → 170-190。
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

    # ========== Component A: goal_proximity (v9: 恢复 v2 的 exp 状态值框架) ==========
    # 距离改进量：每步靠近量，总量 bounded ≈ initial_dist - final_dist
    dist_improvement = current_dist - next_dist

    # v9: 恢复 v2 验证有效的指数邻近奖励（状态值信号）。
    # exp(-dist/0.5): dist=0→1.0, dist=0.5→0.37, dist=1.0→0.14, dist=2.0→0.02
    # 这是让 agent 能通过随机探索发现"靠近目标=好"的关键信号——
    # 不需要主动朝目标移动，只要碰巧处于较近距离就能获得正向反馈。
    # v5 的悬停 exploit 是因为 landing 被削弱（1.0/腿），
    # 当前 landing=2.0/腿（dual=4.0/步）对 proximity=1.0/步 形成 4:1 压制，安全。
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))

    goal_proximity = 3.0 * dist_improvement + proximity_bonus

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.5→0.29, dist=1.0→0.17, dist=2.0→0.09
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability（保持 v2 基线框架）==========
    # 基线 0.2 确保远处也有最低速度约束
    # 远处: multiplier≈0.2+0.23=0.43, 近处: multiplier≈0.2+2.5=2.7
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint（保持 v2 基线框架）==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 基线 0.3 确保远处也有最低姿态约束
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact（v2 验证有效，2.0/腿）==========
    # 目标附近 + 低速时，双足着陆 = 4.0/步
    # proximity_bonus 在 dist=0 为 1.0/步，landing:proximity = 4:1，压制悬停
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

# 上一轮训练反馈 (score=-114.629125)
# Training Feedback

## Final-policy outcome
score=-114.629125, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.582396, -95.059083]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 16.740101 | 43.1% | 45.6% | 100.0% |
| settling_stability | -17.262093 | -44.5% | 44.5% | 100.0% |
| landing_contact | 2.800000 | 7.2% | 7.2% | 1.2% |
| attitude_constraint | -0.902184 | -2.3% | 2.3% | 100.0% |
| fuel_efficiency | -0.137500 | -0.4% | 0.4% | 4.0% |

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
| -117.0 | -117.4 | -122.5 | -124.3 | -114.0 | -119.3 | -117.3 | -114.7 | -116.8 | -114.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -3.884 | -3.928 | -3.035 | -2.041 | -1.866 | -1.456 | -1.188 | -1.051 | -1.249 | -0.904 |
| fuel_efficiency | -0.318 | -0.165 | -0.113 | -0.148 | -0.263 | -0.143 | -0.130 | -0.140 | -0.128 | -0.145 |
| goal_proximity | 16.657 | 16.647 | 16.604 | 16.632 | 16.633 | 16.684 | 16.688 | 16.710 | 16.700 | 16.743 |
| landing_contact | 4.800 | 4.800 | 2.700 | 2.500 | 2.500 | 2.400 | 2.500 | 2.700 | 2.400 | 2.600 |
| settling_stability | -17.498 | -17.509 | -17.365 | -17.319 | -17.310 | -17.296 | -17.355 | -17.356 | -17.298 | -17.261 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 9.1% | 4.7% | 3.3% | 4.3% | 7.7% | 4.2% | 3.8% | 4.1% | 3.7% | 4.2% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 2.7% | 2.7% | 1.2% | 1.0% | 1.0% | 1.0% | 1.0% | 1.2% | 1.0% | 1.2% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v8。

    v7→v8 修改（L2 结构性改动 — 单一语义：用速度方向奖励替代状态邻近奖励）：

    问题诊断（v2-v7 全系列）：
    v2-v6 共享 `dist_improvement + coeff*exp(-dist/tau)` 框架，存在内在矛盾：
    - coeff 大 (v2=1.0)：导航有效但悬停可收割（v5 暴露，1000步截断）
    - coeff 小 (v6=0.3)：悬停消除但导航梯度死亡（68步 crash）
    v7 的 potential-based shaping `5.0/(0.2+d)` 在远处梯度也趋于零 → 68步 crash。

    根本原因：所有历史版本都用「距离状态的函数」作为导航奖励，
    这不可避免地在远处梯度弱化（指数/倒数衰减）。

    v8 方案：用「速度在目标方向上的投影」替代 exp(-dist/tau)。
    - vel_toward = -(v · pos) / |pos|：速度向量在指向原点方向上的投影
    - 正值 = 正在靠近目标，负值 = 正在远离
    - 梯度均匀（不随距离衰减）：dist=5 和 dist=0.5 处同样速度产生同样奖励
    - 悬停免疫：hovering → v≈0 → vel_toward≈0，零收割
    - 总量天然 bounded：总奖励 ≈ 系数 × 总路径长度，与 episode 长度无关

    其余组件保持 v2 验证有效的基线框架（settling/attitude 带 baseline，
    landing per-step 2.0/腿），因为 v2 证明这些设置可以完成任务。
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

    # ========== Component A: goal_proximity (v8: delta + velocity-direction) ==========
    # 距离改进量：每步靠近量，总量 bounded ≈ initial_dist - final_dist
    dist_improvement = current_dist - next_dist

    # 速度在目标方向上的投影（v8 核心创新）
    # 单位方向向量 = (-pos_x, -pos_y) / |pos|，指向原点（目标）
    # vel_toward = dot(velocity, direction_to_target)
    #            = (v_x, v_y) · (-pos_x, -pos_y) / |pos|
    #            = -(v_x*pos_x + v_y*pos_y) / |pos|
    # 正值 → 正在靠近目标；负值 → 正在远离；≈0 → 横向运动或静止
    # 关键性质：梯度不随距离衰减，悬停时 ≈0（无收割可能）
    if next_dist > 0.01:
        vel_toward = -(next_x_vel * next_x_pos + next_y_vel * next_y_pos) / next_dist
    else:
        vel_toward = 0.0

    # dist_improvement 提供主要导航梯度（每步靠近量）
    # vel_toward 提供辅助方向引导，鼓励直线高效靠近（而非绕弯）
    # 系数 0.3：速度=3 直线靠近时贡献 ≈0.9/步，与 v2 的 exp(0)=1.0 量级相当
    goal_proximity = 3.0 * dist_improvement + 0.3 * vel_toward

    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: dist=0→1.0, dist=0.5→0.29, dist=1.0→0.17, dist=2.0→0.09
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)

    # ========== Component B: settling_stability (保持 v2 基线框架) ==========
    # 基线 0.2 确保远处也有最低速度约束
    # 远处: multiplier≈0.2+0.23=0.43, 近处: multiplier≈0.2+2.5=2.7
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier

    # ========== Component C: attitude_constraint (保持 v2 基线框架) ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)

    # 基线 0.3 确保远处也有最低姿态约束
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)

    # ========== Component D: landing_contact (v2 验证有效) ==========
    # 目标附近 + 低速时，双足着陆 = 4.0/步
    # 由于 goal_proximity 不再有 per-step state reward（悬停时 vel_toward≈0），
    # landing 信号干净无竞争
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

```
