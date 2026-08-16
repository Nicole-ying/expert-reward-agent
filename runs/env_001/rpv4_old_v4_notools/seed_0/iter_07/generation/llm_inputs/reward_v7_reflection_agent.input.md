# Duplicate reward retry
The previous generation duplicated iter 6 (runs/env_001/rpv4_old_v4_notools/seed_0/iter_06/generation/reward_v6.py). Retry 2: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python

```

下面是环境描述：
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

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

# 上一轮训练反馈 (iter_6, score=-103.791054)
# Training Feedback

## Final-policy outcome
score=-103.791054, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-130.084365, -56.747589]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 1348.858343 | 99.2% | 99.2% | 100.0% |
| action_penalty | -10.000000 | -0.7% | 0.7% | 100.0% |
| progress_reward | 0.283382 | 0.0% | 0.0% | 100.0% |

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
| -2556.7 | -1030.2 | -207.5 | -148.9 | -89.6 | -90.4 | -75.1 | -48.2 | -127.5 | -108.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | -3.103 | -10.000 | -9.301 | -10.000 | -9.841 | -10.000 | -9.988 | -9.912 | -10.000 | -10.000 |
| landing_reward | 36.718 | 166.035 | 1117.167 | 1213.178 | 1394.697 | 1381.017 | 1414.677 | 1500.410 | 1291.565 | 1345.998 |
| progress_reward | -3.684 | -5.522 | -0.350 | -0.054 | 0.478 | 0.403 | 0.505 | 0.715 | 0.085 | 0.231 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.9% | 99.1% | 100.0% | 100.0% |
| landing_reward | 46.2% | 41.1% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0          # weight for distance-reduction reward
    c_angle = 2.0             # health gate angle sensitivity
    c_vel = 1.0               # health gate speed sensitivity
    w_land = 2.0              # weight for continuous landing quality reward
    y_threshold = 5.0         # height below which landing quality activates
    speed_scale = 1.0         # speed penalty temperature
    angle_scale = 0.5         # angle penalty temperature
    contact_bonus = 10.0      # bonus for stable two-leg contact
    w_action = 0.01           # weight for action sparsity penalty

    # -------------------------------
    # 1. Goal-reaching progress (distance reduction with health gate)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_raw = dist_cur - dist_next  # positive when getting closer

    health_penalty = c_angle * abs(obs[4]) + c_vel * (abs(obs[2]) + abs(obs[3]))
    health_gate = 1.0 / (1.0 + health_penalty)

    progress_reward = w_progress * progress_raw * health_gate

    # -------------------------------
    # 2. Continuous landing quality (height, speed, angle)
    # -------------------------------
    y = next_obs[1]                          # altitude (assuming ground at 0)
    height_factor = max(0.0, 1.0 - y / y_threshold)

    speed_total = abs(next_obs[2]) + abs(next_obs[3])
    angle_abs = abs(next_obs[4])

    # quality: high when low, slow, and upright
    quality = (
        height_factor *
        (2.718281828 ** (-speed_total / speed_scale)) *
        (2.718281828 ** (-angle_abs / angle_scale))
    )
    landing_reward = w_land * quality

    # bonus for actual stable two-leg contact
    both_legs_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    if both_legs_contact:
        landing_reward += contact_bonus

    # -------------------------------
    # 3. Action sparsity penalty
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + landing_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "landing_reward": landing_reward,
        "action_penalty": action_penalty
    }
    return float(total_reward), components
```

---

以上是本轮数据。现在开始调查，做出一个语义方向上的修改。

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