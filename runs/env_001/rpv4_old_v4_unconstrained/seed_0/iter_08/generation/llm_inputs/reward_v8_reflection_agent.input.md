下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_7, score=83.186927)
# Training Feedback

## Final-policy outcome
score=83.186927, len=766.550000, terminated=8/20, truncated=12/20, reward_errors=0
score_range=[-46.813458, 300.924232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 261.744270 | 81.0% | 81.0% | 5.1% |
| contact_reward | 36.575000 | 11.3% | 11.3% | 5.1% |
| action_reward | -19.433000 | -6.0% | 6.0% | 72.2% |
| progress_reward | 2.350517 | 0.7% | 0.9% | 99.4% |
| vel_damping | -2.206279 | -0.7% | 0.7% | 13.4% |
| orientation_penalty | -0.406323 | -0.1% | 0.1% | 100.0% |

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
| -161.6 | -146.3 | -251.7 | -62.5 | -4.6 | 114.3 | 51.8 | 130.4 | 88.7 | 107.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | -1.740 | -1.117 | -2.144 | -5.847 | -7.571 | -8.929 | -15.774 | -13.212 | -18.623 | -14.695 |
| contact_reward | 1.700 | 1.750 | 0.900 | 20.475 | 31.850 | 95.025 | 72.500 | 131.225 | 22.950 | 31.875 |
| landing_bonus | 1.471 | 1.733 | 0.807 | 45.687 | 69.246 | 303.554 | 224.743 | 669.523 | 150.801 | 218.926 |
| orientation_penalty | -0.696 | -0.603 | -0.624 | -0.578 | -0.578 | -0.794 | -0.793 | -0.692 | -0.460 | -0.450 |
| progress_reward | 1.511 | 1.249 | -0.259 | 1.171 | 1.493 | 2.022 | 2.428 | 2.522 | 2.389 | 2.407 |
| vel_damping | -7.982 | -8.090 | -5.363 | -4.930 | -3.742 | -3.552 | -3.555 | -2.309 | -2.430 | -2.403 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 57.4% | 37.4% | 57.6% | 63.9% | 63.1% | 72.5% | 65.8% | 60.1% | 72.9% | 72.4% |
| contact_reward | 2.7% | 2.6% | 1.1% | 9.9% | 11.0% | 25.9% | 13.8% | 21.9% | 3.5% | 6.1% |
| landing_bonus | 0.9% | 0.6% | 0.2% | 7.0% | 6.6% | 22.8% | 13.2% | 21.9% | 3.5% | 6.1% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 99.2% | 98.9% | 100.0% | 98.9% | 99.2% | 98.9% |
| vel_damping | 91.5% | 96.2% | 95.1% | 64.3% | 42.1% | 29.1% | 16.8% | 16.7% | 14.9% | 17.8% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v7: Continuous landing quality — eliminate binary cliffs that cause policy collapse.

    Root cause of v4/v6 crashes: landing_bonus was a sparse/semi-sparse spike (50/step
    or 200-transition) that dominated the reward landscape. Agent overfit to triggering
    it, then the brittle policy collapsed under continued exploration.

    Fix: Replace binary landing check with a continuous 0–1 quality score.
    - Product of 4 sub-scores: dist, angle, speed, contact — all must be good for high reward
    - Max 12.0/step (vs 50 in v4, 200+5 in v6) — balanced against other components
    - Only activates when at least one foot touches ground (no hovering credit)
    - Contact reward (0.5/foot) kept for ground-discovery gradient
    - Progress boosted to 2.0 to compensate for reduced landing signal
    - All other components at v4-proven scales
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    px = obs[0]
    py = obs[1]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (slightly boosted for navigation gradient) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Orientation penalty (v4 scale: proven stable) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4 scale: sigma=0.5, coeff=-0.5) ──
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: modest ground-discovery incentive ──
    contact_reward = 0.5 * (left_contact + right_contact)

    # ── 5. Continuous landing quality (replaces binary landing_bonus) ──
    any_contact = (left_contact > 0.5 or right_contact > 0.5)

    if any_contact:
        # Each sub-score: 0 (worst) to 1 (perfect), linear decay outside target zone
        dist_quality = max(0.0, 1.0 - dist / 0.5)          # 1 at pad center, 0 at dist>=0.5
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.4)   # 1 when upright, 0 at |angle|>=0.4
        speed_quality = max(0.0, 1.0 - speed / 0.5)        # 1 when still, 0 at speed>=0.5
        contact_quality = (left_contact + right_contact) / 2.0  # 0, 0.5, or 1.0

        # Product: ALL factors matter — partial credit for partial progress
        landing_quality = dist_quality * angle_quality * speed_quality * contact_quality
        landing_bonus = 12.0 * landing_quality
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (v4 scale) ──
    if action == 0:
        action_reward = 0.0
    elif action == 2:          # main engine — highest cost
        action_reward = -0.04
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.02

    # ── Aggregate ──
    total_reward = (
        progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "vel_damping": vel_damping,
        "contact_reward": contact_reward,
        "landing_bonus": landing_bonus,
        "action_reward": action_reward,
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