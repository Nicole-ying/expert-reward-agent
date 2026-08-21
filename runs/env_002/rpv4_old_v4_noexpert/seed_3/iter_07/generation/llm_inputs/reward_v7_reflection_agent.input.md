# Duplicate reward retry
The previous generation duplicated iter 2 (runs/env_002/rpv4_old_v4_noexpert/seed_3/iter_02/generation/reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v7 reward: L1 回退 forward 系数 1.75 -> 1.5，恢复 iter2 验证过的精确配置。
    #
    # 证据：iter2（forward=1.5 + 平滑二次 balance）是历史最佳 285.19，checkpoint
    # cp70 后单调上升、无振荡。iter3（2.0）checkpoint 到 300 但 final 脆、出现摔倒。
    # iter6 仅把 forward 提到 1.75，训练在「快而摔 / 慢而稳」两个盆地间剧烈震荡
    # （checkpoint score: 114.9 -> 183.4 -> -80.1 -> -89.0 -> -34.9 -> -84.7 -> 173.7 -> 90.7），
    # final score 掉到 120.03，score_range=[-95.3, 299.0] 呈双峰，多数种子摔倒。
    #
    # 本轮 L1（单组件）：只把 forward 系数恢复到验证过的 1.5，balance 保持 iter2
    # 平滑二次形不变，让速度/稳定性权衡点回到稳定盆地。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.5 * next_obs[2]

    # 平滑连续平衡正则（iter2 精确形态）：二次基座，全程有梯度，无死区、无悬崖。
    tilt_penalty = 1.0 * next_obs[0] ** 2
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_penalty
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
    }

    return float(total_reward), components
```

下面是环境描述：
本环境是一个匿名 2D 双足运动任务。智能体控制一个两腿躯干，通过髋、膝共 4 个连续力矩动作，在不规则地形上稳定向前行走。任务主目标是尽可能远、尽可能快地前进；必要健康约束是保持身体不倒，因为摔倒会立即终止回合。次要目标是降低能量消耗或关节力矩成本。

该任务不应被理解为纯平衡任务：保持平衡是完成前进目标的条件，而不是最终目标。也不应把“站立不动但保持平衡”当作成功。

目标得分：300

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

# 上一轮训练反馈 (iter_6, score=120.025854)
# Training Feedback

## Final-policy outcome
score=120.025854, len=850.600000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-95.301938, 299.021480]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 592.060783 | 96.5% | 97.2% | 100.0% |
| balance_penalty | -17.108659 | -2.8% | 2.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 1/20
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
| -30.1 | -37.8 | 114.9 | 183.4 | -80.1 | -89.0 | -34.9 | -84.7 | 173.7 | 90.7 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| balance_penalty | -6.761 | -23.292 | -79.230 | -115.612 | -13.287 | -4.518 | -9.910 | -25.195 | -22.063 | -11.721 |
| forward_reward | -7.755 | 11.955 | 449.318 | 653.204 | 77.875 | 47.389 | 206.074 | 67.779 | 685.192 | 513.094 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| balance_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| forward_reward | 96.7% | 96.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v6 reward: L1 微调 forward 系数 1.5 -> 1.75，删除 iter5 的四次尾项。
    #
    # 证据：iter2（forward=1.5 + 1.0*tilt² + 0.2*angvel² + 0.1*vertvel²）是历史最佳 285.19。
    # iter3（forward=2.0）checkpoint cp70~80 达 299.7/302.6，但 final 掉到 281.28，
    # 且 score_range 出现摔倒回合 -> 2.0 太快、最终策略脆弱。
    # iter5 在 iter2 基座上加了 4.0*tilt**4，balance_penalty 在 cp50/cp60 飙到 -189/-206，
    # forward ep_sum 从 756 掉到 606，agent 减速避险，score 掉到 180。
    #
    # 本轮 L1（单组件）：仅把 forward 系数取 1.5 与 2.0 之间的新值 1.75，
    # 同时恢复 iter2 验证过的平滑二次平衡项（不再叠四次尾项），
    # 在不引入新平衡形态的前提下把速度/稳定性权衡点推向略快。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    forward_reward = 1.75 * next_obs[2]

    # 平滑连续平衡正则（iter2 精确形态）：二次基座，全程有梯度，无死区、无悬崖。
    tilt_penalty = 1.0 * next_obs[0] ** 2
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    balance_penalty = (
        tilt_penalty
        + angular_velocity_penalty
        + vertical_velocity_penalty
    )

    total_reward = forward_reward - balance_penalty

    components = {
        "forward_reward": forward_reward,
        "balance_penalty": -balance_penalty,
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