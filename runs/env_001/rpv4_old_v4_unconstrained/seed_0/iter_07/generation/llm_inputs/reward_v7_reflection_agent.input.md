下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_6, score=39.147304)
# Training Feedback

## Final-policy outcome
score=39.147304, len=655.550000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-67.311211, 278.404267]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 183.600000 | 78.6% | 78.6% | 1.9% |
| contact_reward | 26.662500 | 11.4% | 11.4% | 3.5% |
| action_reward | -15.787000 | -6.8% | 6.8% | 65.7% |
| vel_damping | -4.671777 | -2.0% | 2.0% | 21.2% |
| progress_reward | 1.671110 | 0.7% | 1.1% | 99.4% |
| orientation_penalty | -0.389096 | -0.2% | 0.2% | 100.0% |

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
| -113.3 | -103.1 | -111.6 | -51.3 | -51.1 | -1.6 | -4.7 | 102.1 | 146.9 | -0.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | -0.615 | -0.359 | -0.171 | -1.290 | -2.600 | -5.745 | -3.922 | -8.779 | -6.368 | -17.124 |
| contact_reward | 3.000 | 2.625 | 2.550 | 10.537 | 12.037 | 61.425 | 30.150 | 118.650 | 87.900 | 34.163 |
| landing_bonus | 31.100 | 91.500 | 111.400 | 129.000 | 157.250 | 361.800 | 199.500 | 525.100 | 338.650 | 149.400 |
| orientation_penalty | -1.172 | -0.672 | -0.597 | -0.395 | -0.437 | -0.357 | -0.626 | -0.508 | -0.414 | -0.464 |
| progress_reward | 1.663 | 1.677 | 1.682 | 1.655 | 1.624 | 1.586 | 1.723 | 1.634 | 1.655 | 1.587 |
| vel_damping | -10.218 | -10.332 | -10.361 | -9.600 | -8.706 | -6.746 | -9.578 | -5.763 | -4.798 | -6.102 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 33.9% | 22.0% | 12.5% | 44.4% | 50.8% | 59.7% | 53.2% | 58.0% | 56.6% | 67.4% |
| contact_reward | 4.2% | 3.3% | 3.1% | 10.0% | 5.6% | 17.0% | 11.8% | 21.3% | 24.5% | 5.2% |
| landing_bonus | 1.0% | 1.7% | 1.8% | 5.0% | 4.2% | 8.9% | 7.8% | 15.4% | 11.3% | 1.3% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.8% | 99.7% | 99.3% | 98.2% | 99.8% |
| vel_damping | 95.8% | 95.3% | 95.1% | 81.6% | 54.7% | 34.3% | 66.3% | 33.3% | 36.7% | 24.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v6: Restore v4 penalty scales + landing transition detection + modest near-landing bridge.

    Core fixes from v5 regression:
    - ALL penalties restored to v4 levels (vel_damping: -0.5, orientation: -0.1/-0.04, progress: 1.5)
    - Landing bonus uses transition detection: first good-landing step gets +200,
      subsequent sustain gets +5/step — prevents the cp90% 5387 exploit seen in v4 training
    - Near-landing bridge kept at 2/step (not v5's 10) for gradient without dominance
    - Contact reward slightly boosted (0.75) for stronger ground-discovery signal
    - Action costs at v4 levels: -0.04 main, -0.02 side, 0 glide
    """
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Previous observation for transition detection and progress
    px = obs[0]
    py = obs[1]
    pvx = obs[2]
    pvy = obs[3]
    p_angle = obs[4]
    p_left = obs[6]
    p_right = obs[7]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5
    prev_speed = (pvx ** 2 + pvy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (v4 scale: 1.5) ──
    progress_reward = 1.5 * (prev_dist - dist)

    # ── 2. Orientation penalty (v4 scale: -0.1 angle², -0.04 angvel²) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4 scale: narrow gate sigma=0.5, coeff=-0.5) ──
    # Only activates when very close to pad — does not interfere with descent
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward (slightly boosted: 0.75/foot for stronger ground discovery) ──
    contact_sum = left_contact + right_contact
    contact_reward = 0.75 * contact_sum

    # ── 5. Landing bonus with transition detection ──
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow

    # Detect previous landing state from obs (transition detection)
    prev_both = (p_left > 0.5 and p_right > 0.5)
    prev_near = (prev_dist < 0.6)
    prev_upright = (abs(p_angle) < 0.35)
    prev_slow = (prev_speed < 0.5)
    was_good_landing = prev_both and prev_near and prev_upright and prev_slow

    # Near-landing: on pad but not perfectly settled — bridges the gap
    is_near_landing = both_on_ground and near_pad and not is_good_landing

    if is_good_landing and not was_good_landing:
        # First step of successful landing: large one-time bonus
        landing_bonus = 200.0
    elif is_good_landing:
        # Sustaining good landing: modest per-step to encourage staying settled
        landing_bonus = 5.0
    elif is_near_landing:
        # On pad but not yet perfect: small bridge incentive
        landing_bonus = 2.0
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