下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_8, score=-53.830120)
# Training Feedback

## Final-policy outcome
score=-53.830120, len=545.500000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[-184.292759, 191.951763]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 921.516514 | 94.7% | 94.7% | 3.7% |
| contact_reward | 32.380000 | 3.3% | 3.3% | 20.9% |
| action_reward | -14.110000 | -1.5% | 1.5% | 79.2% |
| progress_reward | 1.235492 | 0.1% | 0.2% | 99.9% |
| vel_damping | -1.398807 | -0.1% | 0.1% | 28.3% |
| orientation_penalty | -1.223701 | -0.1% | 0.1% | 100.0% |

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
| -45.9 | -115.2 | -121.0 | -100.7 | -158.0 | -481.7 | -148.2 | -143.7 | -22.5 | -85.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | -1.367 | -0.376 | -0.223 | -0.337 | -2.373 | -5.226 | -7.467 | -8.751 | -14.101 | -14.603 |
| contact_reward | 1.070 | 0.630 | 0.630 | 0.630 | 0.920 | 0.000 | 0.400 | 5.490 | 39.580 | 34.890 |
| landing_bonus | 35.912 | 38.965 | 34.801 | 38.786 | 24.149 | 0.000 | 7.244 | 135.560 | 926.814 | 1188.494 |
| orientation_penalty | -0.529 | -1.000 | -0.926 | -0.614 | -1.296 | -1.874 | -0.544 | -0.512 | -1.365 | -0.949 |
| progress_reward | 2.092 | 2.233 | 2.226 | 2.207 | 1.050 | -2.656 | 0.694 | 0.445 | 1.571 | 1.258 |
| vel_damping | -9.488 | -10.298 | -10.304 | -10.265 | -9.137 | -4.041 | -5.371 | -1.230 | -1.120 | -1.367 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 50.0% | 27.5% | 16.3% | 16.4% | 48.1% | 71.1% | 58.6% | 61.5% | 81.3% | 75.9% |
| contact_reward | 4.2% | 3.1% | 3.2% | 3.1% | 2.5% | 0.0% | 0.4% | 4.7% | 25.5% | 21.1% |
| landing_bonus | 1.2% | 1.5% | 1.5% | 1.5% | 0.6% | 0.0% | 0.1% | 0.6% | 3.8% | 4.0% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 100.0% |
| vel_damping | 96.0% | 95.4% | 95.4% | 95.5% | 96.3% | 93.1% | 75.1% | 46.9% | 22.7% | 25.5% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v8: Sum-based touchdown quality + one-time bonus. Eliminates per-step landing accumulation.

    Root cause of v4/v6/v7 failures:
    - Per-step landing rewards (binary 50/step or product 12/step) accumulate while agent
      sits on ground, inevitably dominating the reward landscape.
    - Product formulation (v7) creates multiplicative cliffs: one bad factor → reward=0.
    - This causes the classic exploit-collapse pattern at cp80% in training.

    Fixes:
    1. PRODUCT → SUM: landing_quality = dist + angle + speed + contact_quality.
       Each factor contributes independently → smooth gradient, no cliffs.
    2. ONE-TIME touchdown bonus: reward is given only at the moment of first ground
       contact (air→ground transition), proportional to landing quality.
       Max ~80, no per-step sustain → cannot dominate through accumulation.
    3. contact_reward reduced to 0.2/foot — retains ground-discovery gradient without
       excessive accumulation.
    4. All other components at v4/v7-proven scales.
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
    p_left = obs[6]
    p_right = obs[7]

    # ── Derived quantities ──
    dist = (x ** 2 + y ** 2) ** 0.5
    prev_dist = (px ** 2 + py ** 2) ** 0.5
    speed = (vx ** 2 + vy ** 2) ** 0.5

    # ── 1. Progress: delta-distance (v7 scale, proven navigation signal) ──
    progress_reward = 2.0 * (prev_dist - dist)

    # ── 2. Orientation penalty (v7 scale: stable, always active) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad (v4/v7 scale: narrow gate, modest coefficient) ──
    proximity_gate = 2.718281828 ** (- (dist ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: modest ground-discovery incentive (reduced from 0.5) ──
    contact_reward = 0.2 * (left_contact + right_contact)

    # ── 5. Landing bonus: ONE-TIME touchdown quality (SUM-based, no per-step sustain) ──
    # Detect air→ground transition
    was_airborne = (p_left < 0.5 and p_right < 0.5)
    touching_now = (left_contact > 0.5 or right_contact > 0.5)
    just_touched_down = was_airborne and touching_now

    if just_touched_down:
        # Each quality factor: 0 (worst) to 1 (perfect), linear decay
        dist_quality = max(0.0, 1.0 - dist / 0.5)           # 1 at center, 0 at dist>=0.5
        angle_quality = max(0.0, 1.0 - abs(angle) / 0.4)    # 1 when upright, 0 at |angle|>=0.4
        speed_quality = max(0.0, 1.0 - speed / 0.5)         # 1 when still, 0 at speed>=0.5
        contact_quality = (left_contact + right_contact) / 2.0  # 0, 0.5, or 1.0

        # SUM: each factor contributes independently — smooth gradient, no cliffs
        td_quality = dist_quality + angle_quality + speed_quality + contact_quality
        # Max td_quality = 4.0 → max bonus = 80.0 (one-time, no accumulation)
        landing_bonus = 20.0 * td_quality
    else:
        landing_bonus = 0.0

    # ── 6. Action efficiency (v4/v7 scale) ──
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