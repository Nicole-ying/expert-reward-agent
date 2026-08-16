# Duplicate reward retry
The previous generation duplicated iter 5 (runs/env_001/rpv4_old_v4_notools/seed_0/iter_05/generation/reward_v1.py). Retry 2: generate a materially different reward function.
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

# 上一轮训练反馈 (iter_5, score=-121.577260)
# Training Feedback

## Final-policy outcome
score=-121.577260, len=68.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-144.398763, -99.764423]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_reward | 1.777383 | 78.4% | 78.4% | 0.8% |
| progress_reward | 0.408980 | 18.0% | 18.9% | 100.0% |
| action_penalty | -0.062500 | -2.8% | 2.8% | 9.1% |

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
| -116.6 | -105.1 | -113.2 | -117.0 | -120.2 | -118.2 | -117.4 | -121.8 | -115.0 | -119.4 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | -0.067 | -0.098 | -0.044 | -0.033 | -0.030 | -0.030 | -0.034 | -0.057 | -0.070 | -0.100 |
| landing_reward | 1.218 | 2.022 | 1.412 | 1.332 | 1.492 | 1.452 | 1.459 | 1.577 | 1.810 | 1.814 |
| progress_reward | 0.403 | 0.419 | 0.415 | 0.414 | 0.410 | 0.412 | 0.414 | 0.409 | 0.425 | 0.420 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| action_penalty | 9.7% | 14.0% | 6.4% | 4.7% | 4.4% | 4.4% | 4.9% | 8.3% | 10.3% | 14.6% |
| landing_reward | 0.7% | 1.0% | 0.9% | 0.9% | 0.9% | 0.8% | 0.8% | 0.7% | 0.8% | 0.9% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # -------------------------------
    # Hyperparameters
    # -------------------------------
    w_progress = 1.0
    c_angle = 2.0      # health gate sensitivity for body angle
    c_vel = 1.0        # health gate sensitivity for total speed
    w_land = 5.0
    angle_thresh = 0.5   # rad, quality for landing
    vel_thresh = 1.0     # m/s (or units), total speed quality for landing
    w_action = 0.01

    # -------------------------------
    # 1. Goal‑reaching progress (improvement_delta)
    # -------------------------------
    dist_cur = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_unscaled = dist_cur - dist_next  # positive when getting closer

    # -------------------------------
    # 2. Soft health gate (on current body state)
    # -------------------------------
    # Use body angle and total speed to scale progress
    health_penalty = c_angle * abs(obs[4]) + c_vel * (abs(obs[2]) + abs(obs[3]))
    health_gate = 1.0 / (1.0 + health_penalty)

    progress_reward = w_progress * progress_unscaled * health_gate

    # -------------------------------
    # 3. Landing quality (joint condition proxy)
    # -------------------------------
    both_legs_contact = 1.0 if (next_obs[6] > 0.5 and next_obs[7] > 0.5) else 0.0
    if both_legs_contact > 0.0:
        angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / angle_thresh)
        speed_total = abs(next_obs[2]) + abs(next_obs[3])
        speed_factor = max(0.0, 1.0 - speed_total / vel_thresh)
        landing_quality = angle_factor * speed_factor
    else:
        landing_quality = 0.0

    landing_reward = w_land * landing_quality

    # -------------------------------
    # 4. Action sparsity
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