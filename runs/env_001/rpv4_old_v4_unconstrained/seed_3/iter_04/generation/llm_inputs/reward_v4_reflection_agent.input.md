下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_3, score=129.705250)
# Training Feedback

## Final-policy outcome
score=129.705250, len=725.300000, terminated=9/20, truncated=11/20, reward_errors=0
score_range=[-21.356691, 291.378473]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 483.390637 | 60.2% | 60.2% | 8.4% |
| landing_approach | 304.570000 | 37.9% | 37.9% | 100.0% |
| goal_proximity | -9.747461 | -1.2% | 1.2% | 100.0% |
| velocity_penalty | -2.798275 | -0.3% | 0.3% | 99.6% |
| fuel_penalty | -1.586550 | -0.2% | 0.2% | 72.9% |
| orientation_penalty | -0.696511 | -0.1% | 0.1% | 100.0% |

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
| -18.7 | -104.7 | -66.8 | 150.9 | 157.0 | 145.9 | 147.1 | 110.7 | 124.5 | 111.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 4.173 | 1.552 | 30.626 | 145.300 | 510.656 | 315.155 | 372.972 | 459.040 | 594.672 | 551.748 |
| fuel_penalty | -1.007 | -1.176 | -1.773 | -1.131 | -1.270 | -1.452 | -1.465 | -2.119 | -2.266 | -2.239 |
| goal_proximity | -16.114 | -20.336 | -24.134 | -16.144 | -12.114 | -12.483 | -15.041 | -13.484 | -12.263 | -10.941 |
| landing_approach | 53.032 | 58.859 | 122.498 | 131.097 | 236.101 | 238.309 | 195.661 | 343.891 | 363.945 | 344.780 |
| orientation_penalty | -0.955 | -0.648 | -0.904 | -1.628 | -0.689 | -0.893 | -1.048 | -1.019 | -0.711 | -0.929 |
| velocity_penalty | -3.632 | -3.002 | -2.508 | -2.254 | -2.308 | -2.393 | -2.476 | -2.696 | -2.858 | -2.795 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 0.2% | 0.1% | 0.7% | 3.3% | 10.1% | 6.0% | 7.4% | 6.6% | 8.5% | 8.4% |
| fuel_penalty | 85.6% | 81.3% | 87.8% | 64.4% | 65.0% | 71.2% | 73.6% | 78.8% | 84.8% | 89.3% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.9% | 100.0% | 100.0% | 99.5% | 99.3% | 99.0% | 99.2% | 99.7% | 99.6% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: one-shot contact bonus (transition-based) + stronger shaping signals

    Core fix: contact_bonus was per-step with w=25, accumulating to 92% of reward.
    Now it fires ONCE on the transition from "no contact" → "contact", with a
    velocity gate that penalises high-speed touchdowns. The continuous shaping
    signals (landing_approach, goal_proximity) are strengthened to fill the gradient
    that contact_bonus previously monopolised.
    """

    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_cur, right_cont_cur = obs[6], obs[7]
    left_cont_nxt, right_cont_nxt = next_obs[6], next_obs[7]

    # --- helpers ---
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2

    # ================================================================
    # 1. goal_proximity — distance penalty + approach incentive
    #    w_approach doubled (0.4 → 0.8) to give stronger gradient
    # ================================================================
    w_dist = 0.08
    proximity_penalty = -w_dist * dist_nxt

    w_approach = 0.8
    delta_dist = dist_cur - dist_nxt          # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — penalise speed, stronger near the pad
    #    w_v increased 0.15→0.20, k_gate 2.0→3.0 for sharper gating
    # ================================================================
    w_v = 0.20
    k_gate = 3.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)    # ~1 when close, ~0 when far
    velocity_penalty = -w_v * gate * speed_sq

    # ================================================================
    # 3. orientation_penalty — unchanged weight, effective
    # ================================================================
    w_ang = 0.03
    w_angvel = 0.005
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ================================================================
    # 4. landing_approach — weight doubled (0.3 → 0.6),
    #    temperature parameters slightly widened for broader basin
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.30)   # was 0.25
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.40)   # was 0.35
    slow     = 2.718281828 ** (-speed_sq / 1.0)      # was 0.8
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.35) # was 0.3
    landing_approach = 0.6 * near_y * near_x * slow * upright

    # ================================================================
    # 5. contact_bonus — ONE-TIME transition bonus (not per-step!)
    #    Detects the moment both legs first touch down.
    #    Velocity gate ensures high-speed impacts get less credit.
    #    w_contact = 8.0 (was 25.0 per-step → now ~8 once per episode).
    # ================================================================
    w_contact = 8.0
    had_contact = left_cont_cur * right_cont_cur       # 0 or 1, were we already touching?
    has_contact = left_cont_nxt * right_cont_nxt       # 0 or 1, are we touching now?
    new_contact = has_contact * (1.0 - had_contact)    # 1 ONLY on the transition step

    spatial_gate = 2.718281828 ** (-(x_nxt ** 2 + y_nxt ** 2) / 0.5)
    soft_gate    = 2.718281828 ** (-speed_sq / 0.5)    # high speed → bonus shrinks
    contact_bonus = w_contact * new_contact * spatial_gate * soft_gate

    # ================================================================
    # 6. fuel_efficiency — unchanged, tiny regularisation
    # ================================================================
    w_fuel = 0.003
    fuel_penalty = -w_fuel * (0.0 if action == 0 else 1.0)

    # ================================================================
    # assemble
    # ================================================================
    total_reward = (
        goal_proximity
        + velocity_penalty
        + orientation_penalty
        + landing_approach
        + contact_bonus
        + fuel_penalty
    )

    components = {
        'goal_proximity': goal_proximity,
        'velocity_penalty': velocity_penalty,
        'orientation_penalty': orientation_penalty,
        'landing_approach': landing_approach,
        'contact_bonus': contact_bonus,
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