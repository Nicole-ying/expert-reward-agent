# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。根据训练反馈，修改奖励函数以改善外部任务表现。

# 可用工具

以下工具随时可用，在诊断过程中主动调用：

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action。
- `get_component_history(component_name)` — 某个组件在所有迭代中的 ep_sum_mean 和 active_rate 演化轨迹。
- `read_training_feedback(iteration)` — 某轮训练的详细反馈：精确分数、每个组件的 ep_sum/active_rate/magnitude_share。
- `read_past_reflection(iteration)` — 你过去某轮的诊断和干预记录。
- `read_reward_code(iteration)` — 某轮的完整奖励函数代码。
- `read_environment_card()` — 查看本环境的目标、观察空间、动作空间、终止条件。
- `read_checkpoint_trend(iteration)` — 查看某轮训练中各组件随训练的演化轨迹。

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

# 上一轮训练反馈 (iter_3, score=-65.002372)
# Training Feedback

## Final-policy outcome
score=-65.002372, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-88.254004, -28.263688]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| goal_proximity | 1204.083455 | 86.0% | 86.0% | 100.0% |
| attitude_constraint | -89.133967 | -6.4% | 6.4% | 100.0% |
| settling_stability | -34.651470 | -2.5% | 4.4% | 100.0% |
| fuel_efficiency | -45.412500 | -3.2% | 3.2% | 90.8% |
| landing_contact | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -993.1 | -1047.6 | -225.5 | -124.4 | -126.2 | -71.7 | -68.0 | -24.3 | -29.3 | -65.3 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -5.166 | -9.592 | -30.227 | -77.167 | -60.265 | -89.244 | -85.926 | -117.230 | -108.393 | -90.308 |
| fuel_efficiency | -8.990 | -25.133 | -30.590 | -30.695 | -31.098 | -31.103 | -32.050 | -32.170 | -36.748 | -42.487 |
| goal_proximity | 22.871 | 76.968 | 469.282 | 771.367 | 742.001 | 1075.637 | 1100.910 | 1625.727 | 1593.727 | 1203.409 |
| landing_contact | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| settling_stability | -48.755 | -52.254 | -31.136 | -27.776 | -26.520 | -30.569 | -32.213 | -46.500 | -46.612 | -34.298 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 29.6% | 15.9% | 59.3% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 88.8% | 70.6% | 61.2% | 61.4% | 62.2% | 62.2% | 64.1% | 64.3% | 73.5% | 85.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| settling_stability | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v3。

    根本性重构（vs v2）：
    - goal_proximity：从「delta + 线性状态惩罚」切换为「有界指数 proximity bonus + delta」。
      指数衰减 e^(-dist) 在 target 处=1.0，远处→0，始终为正且有界，
      提供了可优化的连续梯度，同时不会被量级淹没。
    - settling_stability：大幅降低状态速度惩罚系数（-1.2 vs v2 的 -4.0），
      提升 delta 权重，使远处基本为零、近处温和引导减速。
    - attitude_constraint：门控阈值放宽到 2.0，让 agent 在中距离就开始关注姿态。
    - landing_contact：收紧条件、提高奖励到 25.0，着陆信号更显著。
    """

    # ========== 信号提取 ==========
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    ang_vel = obs[5]

    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    n_angle = next_obs[4]
    n_ang_vel = next_obs[5]
    n_left, n_right = next_obs[6], next_obs[7]

    # L2 距离与速率
    current_dist = (x**2 + y**2) ** 0.5
    next_dist = (nx**2 + ny**2) ** 0.5
    current_speed = (vx**2 + vy**2) ** 0.5
    next_speed = (nvx**2 + nvy**2) ** 0.5

    # ================================================================
    # Component A: goal_proximity
    #   有界指数 proximity bonus（核心创新）+
    #   delta 距离改进（即时方向反馈）
    #
    #   proximity_bonus: 3.0 * e^(-dist)
    #     dist=0.0 → 3.00   dist=0.5 → 1.82   dist=1.0 → 1.10
    #     dist=2.0 → 0.41   dist=5.0 → 0.02
    #   提供了从远处到近处的连续可优化梯度，始终为正且有界。
    # ================================================================
    proximity_bonus = 3.0 * (2.718281828 ** (-next_dist))
    dist_improvement = current_dist - next_dist
    goal_proximity = proximity_bonus + 5.0 * dist_improvement

    # ================================================================
    # Component B: settling_stability
    #   接近度门控的速度管理：远处几乎不干预，近处温和引导减速。
    #   prox_gate: dist=0→1.0, dist=1→0.33, dist=2→0.20, dist=5→0.09
    # ================================================================
    prox_gate = 1.0 / (1.0 + 2.0 * next_dist)
    speed_improvement = current_speed - next_speed

    # delta：奖励减速，近处权重放大
    speed_delta = 3.0 * speed_improvement * (0.5 + 2.0 * prox_gate)

    # 状态惩罚：温和的线性速度惩罚，近处显著、远处可忽略
    speed_state = -1.2 * next_speed * prox_gate

    settling_stability = speed_delta + speed_state

    # ================================================================
    # Component C: attitude_constraint
    #   两级门控：dist>=2.0 完全不干预，dist<=0.5 全力约束，中间线性过渡。
    #   比 v2 的 1.5 阈值更宽松，让 agent 在中距离开始关注姿态。
    # ================================================================
    if next_dist >= 2.0:
        att_gate = 0.0
    elif next_dist <= 0.5:
        att_gate = 1.0
    else:
        att_gate = (2.0 - next_dist) / 1.5  # 从 1.0 线性降到 0.0

    angle_improvement = abs(angle) - abs(n_angle)
    angvel_improvement = abs(ang_vel) - abs(n_ang_vel)

    angle_delta = att_gate * (2.5 * angle_improvement + 1.0 * angvel_improvement)
    angle_penalty = att_gate * (-3.0 * abs(n_angle) - 1.5 * abs(n_ang_vel))

    attitude_constraint = angle_delta + angle_penalty

    # ================================================================
    # Component D: landing_contact
    #   收紧条件（dist<0.25, speed<0.4），提高奖励到 25.0。
    #   双支撑着陆每步大奖励，单边接触小鼓励。
    # ================================================================
    near_target = next_dist < 0.25
    low_speed = next_speed < 0.4
    both_contacts = (n_left > 0.5 and n_right > 0.5)
    any_contact = (n_left > 0.5 or n_right > 0.5)

    landing_contact = 0.0
    if near_target and low_speed:
        if both_contacts:
            landing_contact = 25.0
        elif any_contact:
            landing_contact = 5.0

    # ================================================================
    # Component E: fuel_efficiency
    #   轻量引擎使用惩罚，不阻碍必要的探索。
    # ================================================================
    fuel_efficiency = -0.05 * float(action != 0)

    # ================================================================
    # 汇总
    # ================================================================
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
        'fuel_efficiency': fuel_efficiency,
    }

    return float(total_reward), components
```

---

以上是本轮数据。以下信息可通过工具自主查阅：
- `read_memory()` — 全局记忆表，简单记录每一轮迭代iteration的socre，len长度，奖励函数组件组成
- `read_training_feedback(N)` — 任意迭代的详细训练反馈，了解以前的iteration有过什么好的或者坏的行为，可以避免重蹈覆辙
- `read_reward_code(N)` — 任意迭代的奖励函数代码，方便你深入查看比较好的奖励函数的具体设计
- `read_past_reflection(N)` — 任意迭代的诊断和干预记录，可以查看由好到坏或者由坏到好的记录，吸收学习经验
- `get_component_history(name)` — 组件在所有迭代中的演化，能看清楚某一组件的好的或者坏的演化过程
- `read_environment_card()` — 环境的观察空间、动作空间、终止条件，用来不确定的时候理解每个输入的观测和动作空间的维度信息
- `read_checkpoint_trend(N)` — 训练过程中各组件的趋势，可以看到从训练过程中看出组件是否发挥作用，这样就不仅只看最终反馈，更全面

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

现在开始调查，分析后输出诊断和代码。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。

**2. 完整 Python 代码**（```python ... ``` 代码块）。
```
