# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。你的工作是：根据 user prompt 中的数据和原则，自主查阅历史记录，诊断问题，做出一个语义方向上的修改，输出新代码。

# 可用工具

- `read_memory()` — 全局记忆表：所有迭代的 skeleton、score、best、len、action
- `read_training_feedback(N)` — 某轮训练的详细反馈：分数、每个组件的 ep_sum/active_rate/magnitude_share
- `read_reward_code(N)` — 某轮的完整奖励函数代码
- `read_past_reflection(N)` — 你过去某轮的诊断和干预记录
- `get_component_history(name)` — 某组件在所有迭代中的 ep_sum_mean 和 active_rate 演化
- `read_environment_card()` — 环境信息：任务目标、观察空间（8维）、动作空间（4动作）、终止条件
- `read_checkpoint_trend(N)` — 某轮训练中各组件在不同训练阶段的趋势

# 修改层级

- **单一语义修改**：每轮只解决一个问题。通常是一个组件的修改，但如果需要联动（如改一个组件的公式 + 调另一个组件的权重来配合），只要服务于同一语义也算一次修改。
- **L1**：只调系数/权重，不改数学表达式。适用于组件方向对但力度不对。
- **L2**：改一个组件的数学形态。常见模式参考：

| 证据模式 | 参考变换 | 验证标准 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | 稀疏→连续过程证据 | active_rate 改善，不产生 proxy 徘徊 |
| 极端值支配奖励 | 无界→归一化有界 | 极端轨迹支配下降，方差下降 |
| 占据好状态即可持续获奖 | 状态值→改善量/势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | 全局→阶段门控 | 早期探索与局部约束同时改善 |
| 多个小因子相乘导致塌缩 | 乘积→加性/几何平均 | 非零反馈增多，联合约束保留 |
| 持续事件被重复领取 | 持续→状态转移事件 | 重复积累下降，外部完成保持 |
| proxy 提高但外部任务不升 | 代理→任务完成对齐 | proxy 与外部分数重新同向 |
| 稠密 proxy 形成中分平台 | 全程 proxy→局部/转移任务信号 | 刷新 best，完成相关行为增加 |

- **L3**：换主信号框架。仅当多轮迭代从未达标，且历史所有语义方向都已穷尽。

# 关键规则

- **避免打转**：如果 `read_past_reflection` 显示之前已经改过同一语义且结果不好，换方向。
- **接近目标时优先 L1**：如果当前得分或 best 得分接近目标（差距 < 10%），优先 L1 微调。因为之前的经验发现已多次出现分数接近目标时做 L2，结果越改越差。

# 代码约束

- 禁止使用 `original_reward`、`terminal_success_reward`、`terminal_failure_penalty`。
- 只使用环境事实摘要声明的 obs、next_obs、action、info 字段。
- 函数签名：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回 `(float(total_reward), components)`。
- 平方根用 `** 0.5`，指数用 `2.718281828 ** exponent`。

```

## User Prompt

```markdown
# Duplicate reward retry
The previous generation duplicated iter 2 (runs/env_002/rpv4_old_v4_noexpert/seed_3/iter_02/generation/reward_v2.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v5 reward: 回退 iter4 的悬崖式死区惩罚，恢复 iter2 验证过的平滑连续平衡正则。
    #
    # 证据：iter4 用 40*max(0, |tilt|-0.4)^2 的死区铰链替换平滑二次惩罚后，
    # score 从 285.19 崩到 29.83。checkpoint 显示 cp10 早期探索被重罚(-451)，
    # cp20~40 agent 学会站立不动(forward≈0)来避免越界，最终又慢又摔。
    # iter2 的平滑二次 balance_penalty 与 forward=1.5 是历史最佳组合(285.19)。
    # 本步恢复该组合：连续平衡梯度 + 已验证的最优前向系数。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    # 保留已验证的最优速度系数 1.5（iter2 最佳 285.19）。
    forward_reward = 1.5 * next_obs[2]

    # 平滑连续平衡正则：全程给梯度，无死区、无悬崖。
    # 正常步态倾角小，惩罚轻；越倾斜惩罚越强但连续，不会吓退探索。
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

# 上一轮训练反馈 (iter_4, score=29.828681)
# Training Feedback

## Final-policy outcome
score=29.828681, len=844.750000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[-102.784991, 166.623024]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 288.207456 | 64.4% | 64.4% | 100.0% |
| fall_risk_penalty | -158.047590 | -35.3% | 35.3% | 7.4% |
| stability_penalty | -1.278112 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 10/20
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
| -98.0 | -43.7 | -76.9 | -67.9 | 68.1 | 81.8 | 78.9 | 82.0 | 123.1 | 155.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| fall_risk_penalty | -451.388 | -1.989 | -0.316 | 0.000 | -6.686 | -6.685 | -1.278 | -0.800 | -78.160 | -58.695 |
| forward_reward | 17.368 | 2.256 | -16.073 | 6.852 | 332.049 | 364.097 | 369.136 | 378.622 | 474.600 | 544.146 |
| stability_penalty | -0.149 | -0.125 | -0.067 | -0.177 | -2.883 | -2.948 | -2.612 | -2.559 | -2.295 | -2.372 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| fall_risk_penalty | 56.8% | 1.3% | 0.4% | 0.0% | 7.0% | 6.6% | 1.7% | 1.2% | 2.8% | 4.0% |
| forward_reward | 100.0% | 98.7% | 96.4% | 92.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| stability_penalty | 100.0% | 6.1% | 9.7% | 10.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v4 reward: 主信号仍为前向速度，但把"稳定性"从全局弱平方惩罚
    # 改为"摔倒风险"死区铰链惩罚。
    #
    # 证据：iter3 最终评估 score_range=[-51.748, 300.544] 且
    # early_terminal(<150 且 score<-50)=0/20，说明存在"走得很好但后期摔倒"
    # 的回合（官方摔倒给 -100 terminal penalty，把整轮拉到负），
    # 非摔倒回合均值已约 298.8。缺口在摔倒，不在速度。
    # 此前奖励只有 1.0*hull_angle^2 的弱平滑惩罚，且 info 为空，
    # agent 训练时感知不到官方 -100，缺少临近摔倒的强信号。
    #
    # 本轮 L2：把 tilt 惩罚从"全量程平方"改为"死区铰链平方"，
    # 仅在 |hull_angle| > 0.4 rad 后急剧增强，用观测重构摔倒惩罚梯度。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：前向水平速度。obs[2] / next_obs[2] 约定为前向为正。
    # 保留已验证的最优速度系数 1.5（iter2 最佳 285.19）。
    forward_reward = 1.5 * next_obs[2]

    # 摔倒风险：死区铰链，只惩罚超过 0.4 rad 的躯干倾斜。
    # 正常步态倾角约 0.1~0.15 rad，不会被误伤；临近摔倒时惩罚急剧上升。
    tilt = abs(next_obs[0])
    fall_risk_penalty = 40.0 * max(0.0, tilt - 0.4) ** 2

    # 其余稳定正则：角速度与垂直速度（弱）。
    angular_velocity_penalty = 0.2 * next_obs[1] ** 2
    vertical_velocity_penalty = 0.1 * next_obs[3] ** 2

    stability_penalty = angular_velocity_penalty + vertical_velocity_penalty

    total_reward = forward_reward - fall_risk_penalty - stability_penalty

    components = {
        "forward_reward": forward_reward,
        "fall_risk_penalty": -fall_risk_penalty,
        "stability_penalty": -stability_penalty,
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
```
