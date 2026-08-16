下面是环境描述：
本环境是一个 2D 飞行器/着陆器轨迹优化任务。智能体（飞行器）起始于视口顶部中央附近，需要通过学习控制其主引擎与姿态引擎，**到达并稳定停靠在中央目标平台上**。
**主目标：尽可能快地到达目标平台并实现稳定、安全的接触（着陆）。**
**重要次目标（效率约束）：在完成主目标的过程中，尽量减少引擎推力使用（节约燃料）。**
**不该混淆的目标：** 此任务的核心是导航到达，而非单纯的存活平衡或持续前进。

目标得分：200

### 第一原则：黑盒实验，不是代码重构

你无法预知新奖励函数会引导出什么行为，只能训练完后看结果。
**看到好趋势 → 先理解为什么好，再精准微调。**
大规模重写等于丢弃已有实验证据，重新掷骰子。

**硬约束：如果本轮得分比上一轮改善超过 30 分，你必须先回答：**
上一轮的哪个修改促成了这个改善？证据是什么？
**如果你回答不了，本轮只做 L1，不改数学形态。**

逐组件分析，按以下顺序：

(1) **如果 score 始终接近零或负**，当前奖励函数从根上错了——你必须重写整个 reward，换一个完全不同的主信号框架。

(2) **如果某个组件的值在整个训练过程中几乎不变**（轨迹平坦，checkpoint 之间无差异），说明 RL 无法优化这个组件。考虑：(a) 调整它的系数或温度参数 (b) 改写成不同的数学形式 (c) 丢弃它。

(3) **如果某个组件的量级远大于其他**（magnitude_share 极高），必须把它缩放到合理范围——否则 agent 只被这一个信号驱动。

### 本轮数据

# 上一轮训练反馈 (iter_7, score=237.757371)
# Training Feedback

## Final-policy outcome
score=237.757371, len=517.050000, terminated=14/20, truncated=6/20, reward_errors=0
score_range=[135.690384, 300.934723]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_guidance | 179.747522 | 98.9% | 98.9% | 60.9% |
| orientation_penalty | -1.134041 | -0.6% | 0.6% | 100.0% |
| progress | 0.685672 | 0.4% | 0.4% | 98.2% |
| velocity_penalty | -0.183836 | -0.1% | 0.1% | 97.9% |

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
| -137.0 | -143.8 | 24.7 | -32.5 | 210.3 | 164.6 | 84.2 | 140.1 | 265.9 | 257.5 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.000 | 0.000 | 10.513 | 2.206 | 84.801 | 190.118 | 49.596 | 354.392 | 23.643 | 70.485 |
| orientation_penalty | -0.657 | -0.811 | -2.127 | -1.833 | -1.481 | -1.720 | -2.256 | -1.147 | -1.053 | -1.140 |
| progress | 0.192 | 0.172 | 0.360 | 0.406 | 0.646 | 0.643 | 0.587 | 0.681 | 0.681 | 0.684 |
| velocity_penalty | -0.492 | -0.560 | -0.277 | -0.244 | -0.189 | -0.176 | -0.202 | -0.164 | -0.167 | -0.177 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| contact_guidance | 0.0% | 0.0% | 8.4% | 0.8% | 35.3% | 57.8% | 17.1% | 71.3% | 15.7% | 36.9% |
| orientation_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress | 100.0% | 100.0% | 99.3% | 99.8% | 98.1% | 99.9% | 99.9% | 100.0% | 94.8% | 96.7% |
| velocity_penalty | 100.0% | 100.0% | 99.1% | 99.8% | 97.9% | 99.8% | 99.8% | 100.0% | 94.3% | 96.4% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- distance to target (target at origin (0,0)) ---
    dist_prev = (obs[0]**2 + obs[1]**2) ** 0.5
    dist_next = (next_obs[0]**2 + next_obs[1]**2) ** 0.5
    progress = dist_prev - dist_next                # >0 when moving closer to target

    # --- velocity components ---
    vx = next_obs[2]
    vy = next_obs[3]

    # vertical velocity penalty, amplified when close to the ground (y near 0)
    altitude_factor = 1.0 / (1.0 + 10.0 * abs(next_obs[1]))
    vel_vertical_penalty = -0.02 * (vy ** 2) * altitude_factor

    # horizontal velocity penalty (small constant weight)
    vel_horizontal_penalty = -0.01 * (vx ** 2)

    velocity_penalty = vel_vertical_penalty + vel_horizontal_penalty

    # --- orientation penalty: body angle and angular velocity ---
    orientation_penalty = -0.05 * abs(next_obs[4]) - 0.02 * (next_obs[5] ** 2)

    # --- contact guidance: state-value with comprehensive settling gates ---
    # Reverting from iter_6's transition-based approach (which caused bouncing
    # exploitation: ep_sum=9810, active=12%) back to state-value (iter_1/5).
    #
    # Key L2 change from iter_1/5: the settling gate now checks total speed
    # (vx AND vy, not just |vy|) plus body angle. This closes two exploitation
    # loopholes:
    #   1. Horizontal sliding: iter_1/5 only gated |vy|, so agent could slide
    #      sideways (vx≠0, vy≈0) and collect full contact reward.
    #   2. Tilted stance: iter_1/5 had no angle gate on contact, so agent could
    #      stay tilted on the platform collecting reward without settling.
    #
    # With all three gates (proximity + speed + angle) simultaneously tight,
    # the agent only earns contact reward when it is truly settled — which
    # should trigger body_not_awake_or_settled termination within a few steps.
    contact_val = float(next_obs[6] + next_obs[7])   # 0.0, 1.0, or 2.0
    proximity_factor = max(0.0, 1.0 - dist_next / 5.0)
    speed = (vx**2 + vy**2) ** 0.5
    velocity_ok = max(0.0, 1.0 - speed / 0.5)       # speed < 0.5 for any reward
    angle_ok = max(0.0, 1.0 - abs(next_obs[4]) / 0.3)  # |angle| < 0.3 for any reward
    contact_guidance = 0.3 * contact_val * proximity_factor * velocity_ok * angle_ok

    # --- combine all components ---
    # Progress weight=0.5 matches iter_1 (score=171, best ever).
    # Weights >0.5 (iter_4/5) caused rushing-and-crashing or lingering.
    total_reward = (0.5 * progress) + velocity_penalty + orientation_penalty + contact_guidance

    components = {
        "progress": 0.5 * progress,
        "velocity_penalty": velocity_penalty,
        "orientation_penalty": orientation_penalty,
        "contact_guidance": contact_guidance
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

现在开始调查，做出一个语义方向上的修改。

### 输出要求

**1. 诊断报告**（在代码之前）：
- `evidence`：支持判断的关键数据和历史记录
- `behavior_diagnosis`：策略当前的学到的行为模式是什么，是不健康的吗？结合 score、episode_length、terminated/truncated 判断。
- `signal_completeness`：必要职责是否完备？缺什么信号？
- `selected_level`：L1 / L2 / L3 及选择理由
- `selected_intervention`：目标组件及具体修改内容
- `falsifiable_hypothesis`：为什么该修预计改善策略？
- `expected_next_round`：下一轮策略学习到的行为模式预计是什么？

**2. 完整 Python 代码**（```python ... ``` 代码块）。