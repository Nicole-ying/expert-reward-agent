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

# 上一轮训练反馈 (iter_7, score=274.390805)
# Training Feedback

## Final-policy outcome
score=274.390805, len=1069.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[33.782973, 305.578804]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 1197.463867 | 99.4% | 99.5% | 100.0% |
| balance_penalty | -6.482423 | -0.5% | 0.5% | 100.0% |

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
| -112.8 | -102.0 | -95.8 | 167.9 | -15.3 | -76.3 | 1.6 | 192.5 | 108.0 | 277.0 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| balance_penalty | -16.573 | -5.089 | -28.311 | -84.274 | -9.137 | -6.084 | -7.063 | -7.564 | -5.514 | -6.091 |
| forward_reward | -21.276 | 10.719 | 45.127 | 901.483 | 391.477 | 126.722 | 445.102 | 1015.158 | 763.641 | 1185.700 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| balance_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| forward_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # v7 reward (L2): 主信号从「无界线性速度」改为「有界饱和速度」，根除高速追逐。
    #
    # 证据：iter6(forward=1.75 线性) magnitude_share=97.2% 支配奖励，checkpoint
    # 在快而摔/慢而稳两盆地间震荡（-30→-38→115→183→-80→-89→-35→-85→174→91），
    # final=120、score_range=[-95,299] 双峰，多数种子高速摔倒。
    # iter3(2.0 线性) cp70~80 达 299.7/302.6 但 final 脆；iter2(1.5 线性) 最稳 285。
    # 线性系数轴 1.0/1.5/1.75/2.0 已穷尽；本轮改主信号形态为 tanh 饱和，
    # 把「越快越好」改为「达到巡航速度后边际收益衰减」，上限约 2.0/step。
    #
    # balance 保持 iter2 验证过的平滑二次形态，不重蹈 iter4 死区铰链 / iter5 四次尾项覆辙。
    #
    # info 为空，不使用；无显式 success/failure 标志，不依赖 terminal flag。
    _ = training_progress

    # 主学习信号：有界前向速度奖励。
    # 健康巡航速度约 0.68 m/s（iter2 最佳 285 的平均速度）。
    # 用 tanh(v/0.7) 使 0~0.7 区间梯度强、超速后边际收益衰减，避免极端速度支配奖励。
    v = next_obs[2]
    v_norm = v / 0.7
    # clamp 归一化输入，避免极端速度导致 exp 溢出
    v_norm = max(-3.0, min(3.0, v_norm))
    e2 = 2.718281828 ** (2.0 * v_norm)
    tanh_v = (e2 - 1.0) / (e2 + 1.0)
    forward_reward = 2.0 * tanh_v

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