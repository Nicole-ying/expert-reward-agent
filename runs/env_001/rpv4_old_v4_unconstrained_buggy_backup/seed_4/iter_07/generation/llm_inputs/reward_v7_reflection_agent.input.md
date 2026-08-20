下面是环境描述：
本环境是一个2D飞行器轨迹优化任务。主体从视口顶部中心附近以随机初始力开始，核心目标是**尽快到达中心的目标垫板并稳定停靠**（位置对齐、速度归零、姿态平稳、安全接触）。次要目标是在满足到达的前提下**最小化引擎推力使用**（节省燃料）。任务不能混淆为纯粹的生存平衡，因为主体必须移动到指定地点，而不是单纯保持不倒。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_6, score=-116.652889)
# Training Feedback

## Final-policy outcome
score=-116.652889, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-143.061344, -97.217550]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| settling_stability | -14.658990 | -51.2% | 51.2% | 99.7% |
| goal_proximity | -7.530878 | -26.3% | 26.3% | 100.0% |
| landing_contact | 4.958485 | 17.3% | 17.3% | 1.2% |
| attitude_constraint | -1.356701 | -4.7% | 4.7% | 100.0% |
| fuel_efficiency | -0.129000 | -0.5% | 0.5% | 6.3% |

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
| -118.9 | -114.9 | -120.1 | -123.8 | -123.7 | -118.3 | -117.3 | -117.3 | -117.2 | -116.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | -2.904 | -2.396 | -2.408 | -2.565 | -2.159 | -1.448 | -1.507 | -1.261 | -1.369 | -1.344 |
| fuel_efficiency | -0.225 | -0.360 | -0.210 | -0.135 | -0.114 | -0.114 | -0.129 | -0.111 | -0.117 | -0.129 |
| goal_proximity | -7.550 | -7.546 | -7.544 | -7.544 | -7.539 | -7.532 | -7.534 | -7.530 | -7.531 | -7.531 |
| landing_contact | 4.628 | 4.628 | 4.326 | 3.925 | 4.005 | 4.354 | 4.964 | 4.396 | 4.412 | 4.601 |
| settling_stability | -14.708 | -14.675 | -14.733 | -14.889 | -15.055 | -14.724 | -14.627 | -14.781 | -14.717 | -14.662 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_constraint | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| fuel_efficiency | 11.0% | 17.5% | 10.2% | 6.6% | 5.6% | 5.6% | 6.3% | 5.4% | 5.7% | 6.3% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_contact | 1.1% | 1.0% | 1.0% | 1.0% | 0.9% | 1.0% | 1.2% | 1.0% | 1.0% | 1.2% |
| settling_stability | 99.7% | 99.6% | 99.6% | 99.8% | 99.9% | 99.8% | 99.7% | 99.8% | 99.8% | 99.7% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    2D飞行器轨迹优化任务奖励函数 v6。

    核心改动：landing_contact 从连续状态奖励 → potential-based 势能差奖励。
    - 定义 landing_potential = prox_factor * speed_factor * contact_frac (范围 0~1)
    - 仅奖励 potential 的正向增量：max(0, Δpotential)
    - 单集最大奖励严格 bounded（coeff × 1.0），彻底防止 exploit
    - 一旦达到完美着陆质量（potential=1.0），无法再获得更多 contact 奖励

    设计原则（继承 v4 的成熟结构）：
    - goal_proximity: 二次距离代价 → 引导接近（远处强、近处弱）
    - settling_stability: 二次速度代价 → 引导减速（高速强、低速弱）
    - landing_contact: 势能差奖励 → 奖励"改善着陆质量"而非"保持状态"
    - attitude_constraint: 姿态代价 → 保持平稳
    - fuel_efficiency: 燃料代价 → 节省推力

    Args:
        obs: 当前观测 (8,)
        action: 当前动作 (0-3)
        next_obs: 下一观测 (8,)
        original_reward: 未使用
        info: 未使用
        training_progress: 未使用
    """

    # ========== 信号提取 ==========
    # 上一状态（用于计算 potential 增量）
    prev_x_pos = obs[0]
    prev_y_pos = obs[1]
    prev_x_vel = obs[2]
    prev_y_vel = obs[3]
    prev_left_contact = obs[6]
    prev_right_contact = obs[7]

    # 当前状态（转移后）
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    body_angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ========== 辅助量 ==========
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    speed = (x_vel ** 2 + y_vel ** 2) ** 0.5

    prev_distance = (prev_x_pos ** 2 + prev_y_pos ** 2) ** 0.5
    prev_speed = (prev_x_vel ** 2 + prev_y_vel ** 2) ** 0.5

    # proximity_gate: 1.0 在目标处，~0.1 在 distance≈3
    proximity_gate = 1.0 / (1.0 + 3.0 * distance)

    # ========== Component A: goal_proximity — 二次距离代价 ==========
    # 梯度 = -0.2 * distance，远处梯度强，近处梯度弱（天然 curriculum）
    goal_proximity = -0.1 * distance ** 2

    # ========== Component B: settling_stability — 二次速度代价 ==========
    # 梯度随 speed 和 proximity 增强，接近目标时减速约束更强
    settling_stability = -0.3 * speed ** 2 * (0.2 + 0.8 * proximity_gate)
    settling_stability += -0.08 * ang_vel ** 2 * (0.2 + 0.8 * proximity_gate)

    # ========== Component C: attitude_constraint — 姿态代价 ==========
    angle_error = abs(body_angle)
    ang_vel_mag = abs(ang_vel)
    # gate 范围：远距离 0.3，近距离 1.0
    attitude_gate = 0.3 + 0.7 * proximity_gate
    attitude_constraint = -0.4 * angle_error * attitude_gate
    attitude_constraint += -0.15 * ang_vel_mag * attitude_gate

    # ========== Component D: landing_contact — 势能差着陆奖励 ==========
    # v6 核心改动：从连续状态奖励改为 potential-based 势能差奖励。
    #
    # 定义 landing_potential ∈ [0, 1]：
    #   potential = prox_factor × speed_factor × contact_frac
    # 其中 prox_factor = 1 - d/0.5, speed_factor = 1 - s/1.0,
    # contact_frac = (left + right) / 2
    #
    # 仅当 next_obs 满足 near_pad + slow_enough 时计算奖励：
    #   reward = coeff × max(0, potential_next - potential_prev)
    #
    # 这天然防止 exploit：一旦 potential 达到 1.0，无法再获得增量。
    # 单集最大总奖励 = coeff × 1.0，严格 bounded。

    near_pad_next = distance < 0.5
    slow_enough_next = speed < 1.0

    if near_pad_next and slow_enough_next:
        # 当前 potential
        prox_factor_next = 1.0 - distance / 0.5       # 1.0@d=0, 0.0@d=0.5
        speed_factor_next = 1.0 - speed / 1.0          # 1.0@s=0, 0.0@s=1.0
        contact_frac_next = (left_contact + right_contact) / 2.0
        potential_next = prox_factor_next * speed_factor_next * contact_frac_next

        # 上一状态 potential（若不在 zone 内则视为 0）
        near_pad_prev = prev_distance < 0.5
        slow_enough_prev = prev_speed < 1.0
        if near_pad_prev and slow_enough_prev:
            prox_factor_prev = 1.0 - prev_distance / 0.5
            speed_factor_prev = 1.0 - prev_speed / 1.0
            contact_frac_prev = (prev_left_contact + prev_right_contact) / 2.0
            potential_prev = prox_factor_prev * speed_factor_prev * contact_frac_prev
        else:
            potential_prev = 0.0

        # 仅奖励改善
        landing_contact = 20.0 * max(0.0, potential_next - potential_prev)
    else:
        landing_contact = 0.0

    # ========== Component E: fuel_efficiency — 燃料代价 ==========
    fuel_efficiency = -0.03 * float(action != 0)

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