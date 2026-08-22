下面是环境描述：
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

目标得分：200

### 本轮数据

# 上一轮训练反馈 (iter_2, score=57.229343)
# Training Feedback

## Final-policy outcome
score=57.229343, len=920.650000, terminated=4/20, truncated=16/20, reward_errors=0
score_range=[-2.451211, 250.390359]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 619.180103 | 99.1% | 99.1% | 48.4% |
| progress | 2.540391 | 0.4% | 0.5% | 100.0% |
| orientation_penalty | -2.068911 | -0.3% | 0.3% | 100.0% |
| velocity_penalty | -0.212533 | -0.0% | 0.0% | 100.0% |

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
| -114.1 | -67.6 | -8.5 | 17.5 | 111.3 | 101.0 | 147.9 | 98.9 | 80.8 | 63.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 5.724 | 23.894 | 92.315 | 165.729 | 286.680 | 422.102 | 270.991 | 351.749 | 430.592 | 551.134 |
| orientation_penalty | -1.246 | -2.662 | -3.232 | -3.679 | -2.536 | -2.959 | -2.244 | -2.181 | -2.315 | -1.943 |
| progress | 0.753 | 1.292 | 1.376 | 1.698 | 2.030 | 2.402 | 2.289 | 2.361 | 2.487 | 2.580 |
| velocity_penalty | -0.657 | -0.599 | -0.558 | -0.287 | -0.248 | -0.215 | -0.217 | -0.201 | -0.197 | -0.218 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 10.6% | 4.7% | 33.9% | 26.4% | 40.7% | 51.7% | 49.3% | 55.7% | 61.5% | 44.2% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.3% | 100.0% | 99.9% | 100.0% |
| velocity_penalty | 99.9% | 100.0% | 99.9% | 99.9% | 99.9% | 99.8% | 99.1% | 100.0% | 99.9% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ── distance to target (target at origin (0,0)) ──
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # ── velocity components ──
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # ── orientation penalty: body angle and angular velocity ──
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # ── contact guidance (transition-based: reward achieving contact, not lingering) ──
    contact_prev = float(obs[6] + obs[7])           # 0.0 .. 2.0
    contact_next = float(next_obs[6] + next_obs[7]) # 0.0 .. 2.0
    contact_gained = max(0.0, contact_next - contact_prev)  # fires only on 0→1 transitions

    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    # total speed must be low for safe contact (both horizontal and vertical)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)

    # (a) small per-step signal to avoid complete sparsity
    contact_dense = 0.02 * contact_next * proximity_factor * velocity_ok
    # (b) transition bonus: reward the act of making new contact
    contact_bonus = 5.0 * contact_gained * proximity_factor * velocity_ok

    contact_guidance = contact_dense + contact_bonus

    # ── combine all components ──
    total_reward = (2.0 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 2.0 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
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