下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=79.792839)
# Training Feedback

## Final-policy outcome
score=79.792839, len=315.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-63.662425, 251.356353]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_bonus | 441.808202 | 92.2% | 92.2% | 5.7% |
| landing_approach | 17.237806 | 3.6% | 3.6% | 100.0% |
| goal_proximity | -13.301012 | -2.8% | 2.8% | 100.0% |
| velocity_penalty | -5.003746 | -1.0% | 1.0% | 98.9% |
| orientation_penalty | -1.315216 | -0.3% | 0.3% | 100.0% |
| fuel_penalty | -0.686400 | -0.1% | 0.1% | 72.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
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
| -125.0 | -95.6 | -60.4 | -28.2 | 12.1 | 8.4 | 97.9 | 75.0 | 101.6 | 67.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 23.014 | 31.791 | 199.358 | 1843.905 | 1990.407 | 1898.517 | 984.998 | 1096.652 | 1064.604 | 630.358 |
| fuel_penalty | -0.206 | -0.151 | -1.236 | -1.800 | -1.720 | -1.138 | -0.712 | -0.490 | -0.653 | -0.732 |
| goal_proximity | -4.879 | -4.903 | -28.621 | -19.571 | -23.985 | -14.517 | -10.821 | -10.579 | -12.213 | -14.444 |
| landing_approach | 0.102 | 0.147 | 11.425 | 55.584 | 35.342 | 30.194 | 26.432 | 12.926 | 16.821 | 16.275 |
| orientation_penalty | -0.326 | -0.313 | -0.508 | -0.996 | -1.597 | -1.214 | -0.861 | -0.900 | -1.197 | -1.510 |
| velocity_penalty | -6.379 | -6.216 | -3.925 | -4.773 | -4.355 | -4.094 | -3.093 | -3.219 | -3.442 | -6.169 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_bonus | 1.7% | 2.2% | 1.6% | 12.6% | 15.4% | 19.8% | 13.7% | 18.7% | 14.8% | 7.6% |
| fuel_penalty | 100.0% | 72.4% | 81.7% | 94.9% | 95.0% | 92.1% | 77.4% | 66.7% | 70.5% | 72.8% |
| goal_proximity | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| landing_approach | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| velocity_penalty | 99.7% | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 98.9% | 98.4% | 98.5% | 98.7% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v2: direct distance penalty + approach bonus + soft-gated contact + landing shaping

    Core idea: "being far from target always costs something" — no more lazy hovering.
    A smooth reward bridge connects "near the pad" → "actually landing".
    """

    # --- unpack ---
    x_cur, y_cur = obs[0], obs[1]
    x_nxt, y_nxt = next_obs[0], next_obs[1]
    vx_nxt, vy_nxt = next_obs[2], next_obs[3]
    angle_nxt = next_obs[4]
    angvel_nxt = next_obs[5]
    left_cont_nxt = next_obs[6]
    right_cont_nxt = next_obs[7]

    # --- helpers ---
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_nxt = (x_nxt ** 2 + y_nxt ** 2) ** 0.5
    speed_sq = vx_nxt ** 2 + vy_nxt ** 2

    # ================================================================
    # 1. goal_proximity — direct penalty + approach incentive
    #    "You pay rent for being away. You earn credit for getting closer."
    # ================================================================
    w_dist = 0.08                     # per-unit-distance penalty
    proximity_penalty = -w_dist * dist_nxt

    w_approach = 0.4                  # bonus per unit of distance reduced
    delta_dist = dist_cur - dist_nxt  # positive = getting closer
    approach_bonus = w_approach * max(0.0, delta_dist)

    goal_proximity = proximity_penalty + approach_bonus

    # ================================================================
    # 2. velocity_damping — penalise speed, stronger near the pad
    # ================================================================
    w_v = 0.15
    k_gate = 2.0
    gate = 1.0 / (1.0 + k_gate * dist_nxt)   # ~1 when close, ~0 when far
    velocity_penalty = -w_v * gate * speed_sq

    # ================================================================
    # 3. orientation_penalty — reduced weight (was 0.1, now 0.03)
    #    Goal: stay upright but don't let this dominate proximity
    # ================================================================
    w_ang = 0.03
    w_angvel = 0.005
    orientation_penalty = -w_ang * abs(angle_nxt) - w_angvel * (angvel_nxt ** 2)

    # ================================================================
    # 4. landing_approach — continuous shaping bridge
    #    Rewards being in a "ready to land" configuration near the pad:
    #    low altitude, centered, slow, upright.
    #    Uses exponential decay — smooth, always > 0, no hard cliffs.
    # ================================================================
    near_y   = 2.718281828 ** (-abs(y_nxt) / 0.25)    # decays as you go higher
    near_x   = 2.718281828 ** (-abs(x_nxt) / 0.35)    # decays with lateral offset
    slow     = 2.718281828 ** (-speed_sq / 0.8)        # decays with speed
    upright  = 2.718281828 ** (-abs(angle_nxt) / 0.3)  # decays with tilt
    landing_approach = 0.3 * near_y * near_x * slow * upright

    # ================================================================
    # 5. contact_bonus — soft-gated, no longer a needle in a haystack
    #    Uses exponential spatial gate instead of hard |x|<0.2 & |y|<0.1.
    #    Contact anywhere gives *some* reward; contact near (0,0) gives full.
    # ================================================================
    w_contact = 25.0
    both_legs = left_cont_nxt * right_cont_nxt          # 0.0 or 1.0
    spatial_gate = 2.718281828 ** (-(x_nxt ** 2 + y_nxt ** 2) / 0.5)
    contact_bonus = w_contact * both_legs * spatial_gate

    # ================================================================
    # 6. fuel_efficiency — tiny nudge toward engine-off when possible
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