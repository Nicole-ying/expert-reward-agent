下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_3, score=-166.291060)
# Training Feedback

## Final-policy outcome
score=-166.291060, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-199.706985, -124.135086]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| proximity_bonus | 768.778240 | 94.3% | 94.3% | 100.0% |
| action_reward | -28.115500 | -3.4% | 5.3% | 100.0% |
| progress_reward | -0.723104 | -0.1% | 0.1% | 100.0% |
| vel_damping | -1.107418 | -0.1% | 0.1% | 3.8% |
| orientation_penalty | -0.831768 | -0.1% | 0.1% | 100.0% |
| contact_reward | 0.000000 | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -2591.4 | -611.4 | -321.7 | -416.7 | -194.8 | -102.5 | -143.9 | -51.0 | -76.2 | -158.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | -18.787 | -20.210 | -15.810 | -28.056 | -28.016 | -28.082 | -28.030 | -27.869 | -27.828 | -27.962 |
| contact_reward | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| landing_bonus | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| orientation_penalty | -1.517 | -0.385 | -0.405 | -1.564 | -0.715 | -1.639 | -2.166 | -1.217 | -0.962 | -0.811 |
| progress_reward | -38.915 | -8.037 | -3.134 | -5.367 | -1.303 | 0.680 | 0.013 | 1.605 | 1.078 | -0.588 |
| proximity_bonus | 121.618 | 216.512 | 325.600 | 535.957 | 709.473 | 917.224 | 836.024 | 1059.468 | 995.260 | 779.748 |
| vel_damping | -2.000 | -1.467 | -1.079 | -1.113 | -1.036 | -1.052 | -1.196 | -1.029 | -1.046 | -1.051 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_reward | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| proximity_bonus | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| vel_damping | 43.5% | 89.1% | 14.7% | 6.3% | 4.1% | 3.7% | 5.5% | 3.2% | 3.4% | 3.7% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v3: Radical rewrite — eliminate suicide incentive.
    
    Core principles:
    - Proximity is ALWAYS POSITIVE (bounded, 1/(1+dist)) — no suicide gradient
    - Progress reward (delta-distance) for efficient navigation
    - Velocity damping only penalizes EXCESS speed near pad (safe_speed threshold)
    - Relaxed landing conditions for discoverability
    - Contact as progress signal, not sparse rarity
    - Modest action/orientation shaping
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    x_prev = obs[0]
    y_prev = obs[1]

    # ── 1. Proximity bonus: always positive, bounded ──
    # Maximum +2.0 at dist=0, decays smoothly, never negative
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    proximity_bonus = 2.0 / (1.0 + dist_curr)

    # ── 2. Progress reward: delta-distance shaping ──
    # Positive when approaching pad, negative when retreating
    # Bounded total: roughly 2 * (initial_dist - final_dist)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 2.0 * (dist_prev - dist_curr)

    # ── 3. Orientation penalty (moderate) ──
    orientation_penalty = -0.15 * (angle ** 2) - 0.06 * (angvel ** 2)

    # ── 4. Velocity damping: only EXCESS speed near pad ──
    speed = (vx ** 2 + vy ** 2) ** 0.5
    # Gaussian gate: ~1.0 at pad, ~0.37 at dist=1.0, ~0.02 at dist=2.0
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 1.0)
    safe_speed = 0.35          # speed below this is free near pad
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -1.0 * proximity_gate * excess_speed

    # ── 5. Contact reward: any foot on ground is progress ──
    contact_sum = left_contact + right_contact
    contact_reward = 1.0 * contact_sum

    # ── 6. Landing success bonus (relaxed thresholds) ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.8)
    upright = (abs(angle) < 0.4)
    slow = (speed < 0.8)
    is_good_landing = both_on_ground and near_pad and upright and slow
    landing_bonus = 10.0 if is_good_landing else 0.0

    # ── 7. Action efficiency ──
    if action == 0:            # glide — small reward for not wasting fuel
        action_reward = 0.02
    elif action == 2:          # main engine — highest cost
        action_reward = -0.06
    else:                      # orientation engines (1, 3) — moderate cost
        action_reward = -0.03

    # ── Aggregate ──
    total_reward = (
        proximity_bonus
        + progress_reward
        + orientation_penalty
        + vel_damping
        + contact_reward
        + landing_bonus
        + action_reward
    )

    components = {
        "proximity_bonus": proximity_bonus,
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