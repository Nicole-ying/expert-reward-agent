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
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_4, score=120.839410)
# Training Feedback

## Final-policy outcome
score=120.839410, len=291.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-104.546529, 267.530685]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 2627.500000 | 97.4% | 97.4% | 18.0% |
| contact_reward | 59.275000 | 2.2% | 2.2% | 22.1% |
| action_reward | -5.524000 | -0.2% | 0.2% | 52.1% |
| vel_damping | -2.258444 | -0.1% | 0.1% | 37.0% |
| progress_reward | 1.721344 | 0.1% | 0.1% | 97.8% |
| orientation_penalty | -0.798858 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 2/20
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
| -46.8 | -90.6 | -79.2 | -107.9 | -248.9 | -145.3 | 49.4 | 25.1 | 64.5 | -1.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | -2.615 | -0.981 | -1.169 | -1.315 | -10.726 | -12.763 | -16.487 | -18.064 | -14.673 | -3.047 |
| contact_reward | 27.575 | 2.500 | 8.000 | 3.450 | 7.900 | 10.425 | 99.775 | 85.100 | 151.000 | 15.250 |
| landing_bonus | 1240.000 | 45.000 | 72.500 | 17.500 | 342.500 | 452.500 | 3990.000 | 3410.000 | 5387.500 | 665.000 |
| orientation_penalty | -0.548 | -0.374 | -0.630 | -0.566 | -0.270 | -0.288 | -0.926 | -0.588 | -0.973 | -1.279 |
| progress_reward | 1.708 | 1.586 | 1.601 | 1.548 | -0.775 | 0.271 | 1.439 | 1.517 | 1.467 | 1.648 |
| vel_damping | -9.130 | -9.986 | -9.686 | -9.995 | -5.024 | -3.282 | -2.169 | -1.695 | -1.594 | -5.198 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_reward | 74.2% | 33.0% | 39.1% | 61.7% | 68.3% | 69.1% | 71.7% | 74.4% | 74.3% | 51.7% |
| contact_reward | 23.0% | 3.6% | 12.0% | 5.1% | 2.0% | 2.3% | 16.0% | 13.1% | 27.2% | 10.5% |
| landing_bonus | 19.2% | 1.0% | 1.5% | 0.4% | 1.6% | 1.8% | 11.4% | 9.4% | 16.9% | 8.3% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 98.7% |
| vel_damping | 59.7% | 87.5% | 83.0% | 92.7% | 37.1% | 17.2% | 15.3% | 11.7% | 16.9% | 70.5% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v4: Eliminate per-step proximity exploitation.

    Core principles:
    - NO per-step location bonus — only delta-distance progress shapes navigation
    - Landing bonus is the LARGEST positive signal, making landing the optimal choice
    - Contact reward encourages ground interaction for discoverability
    - Velocity damping only near pad, for soft landing guidance
    - Action 0 (glide) is neutral — doing nothing forever earns nothing
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

    # ── 1. Progress: delta-distance (THE navigation signal) ──
    # Positive when approaching pad, negative when retreating.
    # No per-step reward for just *being* near the pad.
    dist_curr = (x ** 2 + y ** 2) ** 0.5
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    progress_reward = 1.5 * (dist_prev - dist_curr)

    # ── 2. Orientation penalty (moderate) ──
    orientation_penalty = -0.1 * (angle ** 2) - 0.04 * (angvel ** 2)

    # ── 3. Velocity damping near pad ──
    # Only penalizes excess speed when close to pad
    speed = (vx ** 2 + vy ** 2) ** 0.5
    proximity_gate = 2.718281828 ** (- (dist_curr ** 2) / 0.5)
    safe_speed = 0.3
    excess_speed = max(0.0, speed - safe_speed)
    vel_damping = -0.5 * proximity_gate * excess_speed

    # ── 4. Contact reward: any foot on ground is progress ──
    # Per-step reward encourages the agent to descend and try touching
    contact_sum = left_contact + right_contact
    contact_reward = 0.5 * contact_sum

    # ── 5. Landing success bonus: LARGE sparse reward ──
    # This MUST be the most attractive positive signal
    both_on_ground = (left_contact > 0.5 and right_contact > 0.5)
    near_pad = (dist_curr < 0.6)
    upright = (abs(angle) < 0.35)
    slow = (speed < 0.5)
    is_good_landing = both_on_ground and near_pad and upright and slow
    landing_bonus = 50.0 if is_good_landing else 0.0

    # ── 6. Action efficiency ──
    # Glide (action 0) is neutral — no free lunch for doing nothing
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
```
