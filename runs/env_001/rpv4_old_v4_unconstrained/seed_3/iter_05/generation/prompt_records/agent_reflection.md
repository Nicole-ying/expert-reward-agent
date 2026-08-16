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
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_4, score=10.311966)
# Training Feedback

## Final-policy outcome
score=10.311966, len=788.250000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-66.990115, 192.251014]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_approach | 27.816165 | 48.5% | 48.5% | 100.0% |
| goal_proximity | -19.561549 | -34.1% | 34.1% | 100.0% |
| fuel_penalty | -4.677300 | -8.2% | 8.2% | 98.9% |
| velocity_penalty | -2.945820 | -5.1% | 5.1% | 100.0% |
| orientation_penalty | -1.860451 | -3.2% | 3.2% | 100.0% |
| settled_reward | 0.254703 | 0.4% | 0.4% | 2.5% |
| contact_speed_penalty | -0.180001 | -0.3% | 0.3% | 2.4% |

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
| -40.7 | -90.6 | -137.1 | -55.5 | -28.8 | -28.3 | -32.1 | 3.4 | -21.6 | -16.9 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_speed_penalty | -2.568 | -0.094 | 0.000 | -0.403 | 0.000 | 0.000 | 0.000 | -0.152 | -0.136 | -0.134 |
| fuel_penalty | -0.500 | -3.810 | -2.884 | -5.327 | -5.780 | -5.816 | -5.803 | -5.633 | -5.701 | -5.649 |
| goal_proximity | -5.822 | -22.740 | -22.590 | -36.091 | -23.188 | -20.631 | -25.521 | -20.713 | -23.606 | -22.136 |
| landing_approach | 1.373 | 21.154 | 5.012 | 10.817 | 32.116 | 41.265 | 26.202 | 41.174 | 30.651 | 33.251 |
| orientation_penalty | -1.354 | -1.766 | -1.181 | -2.089 | -1.558 | -1.694 | -1.713 | -2.076 | -2.054 | -1.860 |
| settled_reward | 8.126 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.564 | 0.024 | 0.003 |
| velocity_penalty | -7.960 | -4.021 | -3.511 | -2.637 | -2.388 | -2.689 | -2.087 | -2.686 | -2.541 | -2.673 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_speed_penalty | 4.2% | 6.5% | 0.0% | 0.6% | 0.0% | 0.0% | 0.0% | 2.1% | 0.8% | 0.2% |
| fuel_penalty | 62.4% | 84.3% | 87.6% | 91.3% | 96.3% | 96.9% | 96.7% | 97.8% | 99.1% | 99.0% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| settled_reward | 4.3% | 6.5% | 0.0% | 0.6% | 0.0% | 0.0% | 0.0% | 2.2% | 0.8% | 0.2% |
| velocity_penalty | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: eliminate bounce-exploitable contact_bonus; replace with per-step settled_reward.
    
    Core changes:
    - REMOVED transition-based contact_bonus (bounce-exploitable).
    - ADDED settled_reward: per-step, both legs must STAY in contact, tight gates.
    - ADDED contact_speed_penalty: speed while touching ground is punished (anti-bounce).
    - REDUCED landing_approach weight 0.6 → 0.20 (anti-hover).
    - ADDED downward-velocity penalty near ground (anti-slam).
    - REBALANCED goal_proximity for cleaner signal separation.
    """

    # --- unpack next_obs ---
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_cont = next_obs[6]
    right_cont = next_obs[7]

    # --- unpack obs (for delta computations) ---
    x_cur = obs[0]
    y_cur = obs[1]

    # --- derived quantities ---
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    speed_sq = vx ** 2 + vy ** 2
    both_legs = left_cont * right_cont          # 0.0 or 1.0

    # ================================================================
    # 1. goal_proximity — distance penalty + approach incentive
    #    Moderate weights. landing_approach handles fine positioning.
    # ================================================================
    w_dist = 0.06
    proximity_penalty = -w_dist * dist

    w_approach = 0.30
    delta_dist = dist_cur - dist                 # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — general speed penalty + anti-slam near ground
    #    w_v increased, plus dedicated downward-velocity penalty.
    # ================================================================
    w_v = 0.28
    k_gate = 5.0
    gate = 1.0 / (1.0 + k_gate * dist)           # ~1 near pad, ~0 far away

    # anti-slam: heavy penalty for downward velocity when very near ground
    w_slam = 1.0
    slam_gate = 1.0 / (1.0 + 20.0 * dist)        # ultra-tight: only |dist|<0.1 matters
    downward_v = max(0.0, -vy)                   # positive = moving downward

    velocity_penalty = (
        -w_v * gate * speed_sq
        - w_slam * slam_gate * (downward_v ** 2)
    )

    # ================================================================
    # 3. orientation_penalty — unchanged structure, slightly higher weights
    # ================================================================
    w_ang = 0.05
    w_angvel = 0.01
    orientation_penalty = -w_ang * abs(angle) - w_angvel * (angvel_nxt := angvel) ** 2

    # ================================================================
    # 4. landing_approach — continuous shaping for pre-contact guidance
    #    Weight reduced 0.6 → 0.20. Temperature tightened slightly.
    #    Provides a broad basin of attraction toward landing configuration.
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y) / 0.22)
    near_x   = 2.718281828 ** (-abs(x) / 0.30)
    slow     = 2.718281828 ** (-speed_sq / 0.45)
    upright  = 2.718281828 ** (-abs(angle) / 0.28)
    landing_approach = 0.20 * near_y * near_x * slow * upright

    # ================================================================
    # 5. settled_reward — per-step reward for STAYING landed properly
    #    Replaces old contact_bonus entirely. Only pays when:
    #    - both legs are ON the ground (sustained, not transition)
    #    - VERY close to (0,0) in position
    #    - VERY low speed
    #    - near-upright orientation
    #    Tight gates mean bouncing → near-zero reward (no re-trigger exploit).
    # ================================================================
    w_settled = 3.5
    s_spatial  = 2.718281828 ** (-(x ** 2 + y ** 2) / 0.04)
    s_slow     = 2.718281828 ** (-speed_sq / 0.03)
    s_upright  = 2.718281828 ** (-abs(angle) / 0.10)
    settled_reward = w_settled * both_legs * s_spatial * s_slow * s_upright

    # ================================================================
    # 6. contact_speed_penalty — anti-bounce: speed while in contact is BAD
    #    Directly punishes the bouncing exploit. Strong enough to matter.
    # ================================================================
    w_contact_spd = 2.0
    contact_speed_penalty = -w_contact_spd * both_legs * speed_sq

    # ================================================================
    # 7. fuel_efficiency — moderate regularisation
    # ================================================================
    w_fuel = 0.006
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + settled_reward
        + contact_speed_penalty
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'settled_reward': settled_reward,
        'contact_speed_penalty': contact_speed_penalty,
        'fuel_penalty': fuel_penalty,
    }

    return total_reward, components
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
