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
下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (score=178.696817)
# Training Feedback

## Final-policy outcome
score=178.696817, len=404.500000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-97.916855, 256.588882]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_contact | 285.600000 | 54.0% | 54.0% | 21.9% |
| goal_proximity | 193.003749 | 36.5% | 36.5% | 100.0% |
| settling_stability | -23.303586 | -4.4% | 4.4% | 99.8% |
| fuel_efficiency | -16.750000 | -3.2% | 3.2% | 82.8% |
| attitude_constraint | -10.678825 | -2.0% | 2.0% | 100.0% |

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
| -60.4 | -99.8 | -90.3 | -58.8 | 23.9 | -96.5 | 117.7 | -35.1 | -16.8 | 206.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -28.215 | -29.908 | -15.745 | -19.186 | -14.193 | -13.209 | -14.727 | -11.347 | -12.334 | -12.394 |
| fuel_efficiency | -28.220 | -28.743 | -23.418 | -27.673 | -32.085 | -32.008 | -36.095 | -34.495 | -37.755 | -15.500 |
| goal_proximity | 334.840 | 297.301 | 311.464 | 341.834 | 237.765 | 280.373 | 515.512 | 310.791 | 359.291 | 186.116 |
| landing_contact | 6.200 | 0.000 | 2.800 | 0.400 | 81.100 | 0.000 | 824.500 | 123.100 | 62.100 | 318.100 |
| settling_stability | -44.030 | -50.812 | -38.989 | -30.446 | -29.974 | -27.974 | -30.689 | -29.961 | -30.240 | -22.194 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 62.7% | 62.4% | 63.9% | 62.3% | 83.9% | 72.6% | 89.5% | 93.0% | 95.7% | 80.6% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.3% | 0.0% | 0.1% | 0.0% | 3.2% | 0.0% | 29.9% | 4.6% | 2.6% | 26.2% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 99.9% | 100.0% | 100.0% | 99.8% |
# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v2。
    核心改进：混合信号框架 — improvement delta + 直接状态评估。
    
    问题诊断（v1）：纯 improvement-delta 导致每 episode 总奖励被初始/终止状态锁死，
    agent 无法通过改进策略影响总奖励，所有组件无学习趋势。
    
    v2 方案：
    - goal_proximity: 保留 delta + 新增指数邻近奖励（创建"漏斗"效应）
    - settling_stability: 改为直接速度惩罚 × 邻近门控
    - attitude_constraint: 改为直接角度惩罚 × 邻近门控
    - landing_contact: 放宽门控条件，提升奖励幅度
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
    
    # 指数邻近奖励：在目标附近提供每步强梯度
    # exp(-dist/0.5): dist=0→1.0, dist=0.5→0.37, dist=1.0→0.14, dist=2.0→0.02
    proximity_bonus = 1.0 * (2.718281828 ** (-next_dist / 0.5))
    
    goal_proximity = 3.0 * dist_improvement + proximity_bonus
    
    # ========== 邻近门控（多个组件共用）==========
    # prox_gate: 在目标处≈1.0，远处≈0.1
    prox_gate = 1.0 / (1.0 + 5.0 * next_dist)
    
    # ========== Component B: settling_stability ==========
    # 直接速度惩罚，邻近目标时大幅增强
    # 远处: ~0.2+0.23=0.43 倍，近处: ~0.2+2.5=2.7 倍
    settling_multiplier = 0.2 + 2.5 * prox_gate
    settling_stability = -0.3 * next_speed * settling_multiplier
    
    # ========== Component C: attitude_constraint ==========
    next_angle_err = abs(next_body_angle)
    next_ang_vel_mag = abs(next_ang_vel)
    
    # 直接角度+角速度惩罚，邻近目标时增强
    attitude_multiplier = 0.3 + 2.0 * prox_gate
    attitude_constraint = -attitude_multiplier * (0.3 * next_angle_err + 0.1 * next_ang_vel_mag)
    
    # ========== Component D: landing_contact ==========
    # 放宽门控条件，提高可触发率
    near_target = next_dist < 0.5
    low_speed = next_speed < 1.0
    
    landing_contact = 0.0
    if near_target and low_speed:
        # 双接触给满分，单接触给一半
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
