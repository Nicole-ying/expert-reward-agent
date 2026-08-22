# Duplicate reward retry
The previous generation duplicated iter 7 (runs/env_001/rpv4_old_v4_notools/seed_0/iter_07/generation/reward_v7.py). Retry 2: generate a materially different reward function.
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

# 上一轮训练反馈 (iter_7, score=-120.095465)
# Training Feedback

## Final-policy outcome
score=-120.095465, len=76.900000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-183.306848, -7.740426]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact_reward | 42.500000 | 91.0% | 91.0% | 1.1% |
| descent_reward | 2.868708 | 6.1% | 6.1% | 86.1% |
| progress_reward | 1.037989 | 2.2% | 2.6% | 100.0% |
| action_penalty | -0.087000 | -0.2% | 0.2% | 11.3% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 17/20
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
| -111.8 | -113.4 | -123.6 | -118.7 | -105.0 | -113.1 | -116.8 | -120.4 | -121.8 | -116.1 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | -0.686 | -0.351 | -0.055 | -0.074 | -0.073 | -0.297 | -0.066 | -0.038 | -0.084 | -0.036 |
| contact_reward | 35.000 | 42.500 | 32.500 | 37.500 | 37.500 | 40.000 | 37.500 | 30.000 | 37.500 | 37.500 |
| descent_reward | 2.864 | 2.870 | 2.858 | 2.866 | 2.866 | 2.869 | 2.865 | 2.855 | 2.872 | 2.863 |
| progress_reward | 1.113 | 1.122 | 1.113 | 1.118 | 1.113 | 1.120 | 1.118 | 1.113 | 1.058 | 1.115 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | 100.0% | 51.2% | 8.0% | 10.9% | 10.5% | 43.3% | 9.6% | 5.5% | 10.9% | 5.2% |
| contact_reward | 1.0% | 1.2% | 1.0% | 1.1% | 1.1% | 1.2% | 1.1% | 0.9% | 1.0% | 1.1% |
| descent_reward | 94.3% | 94.3% | 94.3% | 94.3% | 94.4% | 94.3% | 94.3% | 94.3% | 87.1% | 94.4% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0       # weight for horizontal distance reduction
    w_descent = 2.0        # weight for height descent reward
    contact_bonus = 50.0   # one-time reward for first dual-leg contact event
    w_action = 0.01        # action sparsity penalty weight

    # -------------------------------
    # 1. Horizontal progress (distance to target, no health gate)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_reward = w_progress * (dist_cur - dist_next)

    # -------------------------------
    # 2. Descent reward (altitude reduction only)
    # -------------------------------
    descent_reward = w_descent * max(0.0, obs[1] - next_obs[1])

    # -------------------------------
    # 3. First dual-leg contact event reward
    # -------------------------------
    prev_contact = (obs[6] > 0.5 and obs[7] > 0.5)
    curr_contact = (next_obs[6] > 0.5 and next_obs[7] > 0.5)
    contact_reward = 0.0
    if (not prev_contact) and curr_contact:
        contact_reward = contact_bonus

    # -------------------------------
    # 4. Action sparsity penalty
    # -------------------------------
    action_penalty = -w_action * (1.0 if action != 0 else 0.0)

    # -------------------------------
    # Total reward
    # -------------------------------
    total_reward = progress_reward + descent_reward + contact_reward + action_penalty
    components = {
        "progress_reward": progress_reward,
        "descent_reward": descent_reward,
        "contact_reward": contact_reward,
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