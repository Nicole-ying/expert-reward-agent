下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_6, score=-124.571050)
# Training Feedback

## Final-policy outcome
score=-124.571050, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-148.794243, -104.867050]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 13.372057 | 65.6% | 68.0% | 100.0% |
| soft_approach_reward | 4.742454 | 23.3% | 23.3% | 100.0% |
| orientation_penalty | -1.286041 | -6.3% | 6.3% | 100.0% |
| landing_bonus | 0.390868 | 1.9% | 1.9% | 1.5% |
| action_penalty | -0.110000 | -0.5% | 0.5% | 3.2% |

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
| -114.9 | -117.2 | -122.8 | -124.1 | -122.9 | -125.9 | -127.6 | -120.9 | -123.6 | -124.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | -0.082 | -0.168 | -0.080 | -0.103 | -0.100 | -0.088 | -0.070 | -0.108 | -0.113 | -0.120 |
| landing_bonus | 0.374 | 0.391 | 0.374 | 0.437 | 0.419 | 0.443 | 0.400 | 0.489 | 0.417 | 0.416 |
| orientation_penalty | -2.129 | -2.220 | -1.490 | -1.325 | -1.435 | -1.243 | -1.364 | -1.036 | -1.266 | -1.305 |
| progress_reward | 13.370 | 13.505 | 13.359 | 13.362 | 13.367 | 13.368 | 13.353 | 13.391 | 13.369 | 13.373 |
| soft_approach_reward | 5.166 | 5.084 | 4.795 | 4.731 | 4.800 | 4.751 | 4.734 | 4.796 | 4.742 | 4.740 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | 2.4% | 4.8% | 2.3% | 3.0% | 2.9% | 2.6% | 2.0% | 3.1% | 3.3% | 3.5% |
| landing_bonus | 1.5% | 1.6% | 1.5% | 1.5% | 1.5% | 1.5% | 1.5% | 1.5% | 1.5% | 1.5% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| soft_approach_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v6: Replace punitive vel_damping with positive soft_approach_reward
        (reward low speed near pad instead of penalising speed);
        restore landing_bonus to moderate weight + moderate quality gates
        (between iter_4's loose/exploitable and iter_5's tight/dead);
        differentiate full landing vs partial contact, anti-bounce via first-contact gate.
    """
    # ── Extract signals ──────────────────────────────────────────────
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]

    # ── 1. Progress reward (potential-based: bounded total, no hover exploit) ─
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # >0 when approaching
    w_progress = 12.0
    progress_reward = w_progress * progress

    # ── 2. Orientation penalty ───────────────────────────────────────
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # ── 3. Soft approach reward (POSITIVE reward for low speed near pad) ─
    # Replaces punitive vel_damping: agent earns reward for decelerating,
    # creating a positive gradient instead of a cost to be "tolerated".
    dist_thresh = 3.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 far away, 1 at origin
    speed = (vx ** 2 + vy ** 2) ** 0.5
    w_soft = 0.4
    sigma_speed = 0.6                                    # moderate: rewards meaningful slowing
    soft_approach_reward = (
        w_soft * gate *
        2.718281828 ** (- (speed ** 2) / (2.0 * sigma_speed ** 2))
    )

    # ── 4. Landing contact bonus (transition-based, anti-bounce) ─────
    prev_contacts = (1.0 if obs[6] > 0.5 else 0.0) + (1.0 if obs[7] > 0.5 else 0.0)
    curr_contacts = (1.0 if next_obs[6] > 0.5 else 0.0) + (1.0 if next_obs[7] > 0.5 else 0.0)
    new_contacts = max(0.0, curr_contacts - prev_contacts)

    # Moderate quality gates: between iter_4 (1.5/0.5 → loose, exploitable)
    # and iter_5 (0.8/0.3 → tight, dead).  Graded enough to guide learning
    # but suppress high-speed bounce re-trigger.
    quality = (
        2.718281828 ** (- (dist_curr ** 2) / (2.0 * 1.0 ** 2)) *
        2.718281828 ** (- (speed ** 2)    / (2.0 * 0.4 ** 2)) *
        2.718281828 ** (- (angle ** 2)    / (2.0 * 0.3 ** 2))
    )

    w_landing = 45.0                                     # between 30 (dead) and 80 (exploit)
    # Anti-bounce gate: only reward FIRST contact transition (prev_contacts == 0).
    # Once any leg has touched, subsequent re-contacts earn no bonus.
    # This blocks the iter_4 bounce-exploit while preserving the initial touchdown signal.
    if prev_contacts == 0.0 and new_contacts >= 2.0:
        landing_bonus = w_landing * quality
    elif prev_contacts == 0.0 and new_contacts >= 1.0:
        landing_bonus = 0.3 * w_landing * quality        # partial credit for first single-leg touch
    else:
        landing_bonus = 0.0

    # ── 5. Action efficiency ─────────────────────────────────────────
    if action == 0:                                      # no engine
        action_penalty = 0.0
    elif action == 1 or action == 3:                     # orientation engines
        action_penalty = -0.05
    else:                                                # action == 2, main engine
        action_penalty = -0.2

    # ── Assemble ─────────────────────────────────────────────────────
    total_reward = (
        progress_reward +
        orientation_penalty +
        soft_approach_reward +
        landing_bonus +
        action_penalty
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "soft_approach_reward": soft_approach_reward,
        "landing_bonus": landing_bonus,
        "action_penalty": action_penalty
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