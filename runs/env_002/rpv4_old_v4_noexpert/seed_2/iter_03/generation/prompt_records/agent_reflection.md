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
# ⚠️ 上一版代码验证失败
错误信息：Reward v3 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs/env_002/rpv4_old_v4_noexpert/seed_2/iter_03/generation/validations/reward_v3.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
## 诊断报告

### evidence
- `read_memory()`：iter1 `score=233.56`（best），iter2 加了 `energy_penalty` 后 `score=187.89`，`delta=-45.67`，`len` 从 1092 掉到 803。
- `read_training_feedback(1/2)`：
  - iter1：`forward_progress_reward` ep_sum=412.14，`upright_stability_penalty`=-6.07，len=1092，score=233.56。
  - iter2：`forward_progress_reward` ep_sum=332.11（下降 80），`energy_penalty`=-34.19（magnitude_share 9.2%），len=803，score=187.89。
- **关键规律**：`score / forward_progress_reward_ep_sum` 在两轮里几乎恒定（233.56/412.14=0.567，187.89/332.11=0.566）。说明外部 score 基本只由“前进速度积分”驱动；energy/stability 罚项不直接进入 score，只是通过改变行为（episode 长度、速度）间接影响 score。
- `read_checkpoint_trend(1)`：iter1 的 cp100% score=247.5、forward=426.8；`read_checkpoint_trend(2)`：cp70% 后 forward 平台在 ~354-377，score 平台在 ~186-229。前进速度是当前瓶颈。
- iter2 的 energy 罚项使 episode 变短（1092→803）、forward 积分下降（412→332），是 score 回退的直接原因。

### behavior_diagnosis
- 当前策略学到的是“中等速度、较长存活”的步态：平均每步 forward 分量约 0.38-0.41（对应水平速度 v≈0.41-0.45），len 800-1090，20/20 最终都因摔倒终止。
- 这个步态稳定但太慢：要够到 300 分，`forward_sum` 需约 530（300/0.566），而 iter1 只有 412。在 episode 长度已经 ~1090 的情况下，继续只靠拉长存活很难再涨，**速度是瓶颈**。
- iter2 的能量罚项没有换来更高效步态，反而缩短了存活时间，是不健康的信号分配。

### signal_completeness
- `forward_progress`：职责在，但 `v/sqrt(v²+1)` 在 v>0.4 后边际收益快速衰减，速度出现平台，缺少对“更快前进”的持续梯度。
- `upright_stability`：职责在，但 `angle_weight=1.0` 太弱——在接近摔倒的 1.0 rad 倾斜时每步只罚 0.25，小于一步 forward 收益，无法有效阻止摔倒。
- `energy_efficiency`：iter2 加入后被证明有害，且外部 score 不包含该项，应移除。

### selected_level
- **L1**（只调系数/权重，不改数学表达式）。
- 理由：`forward` 的归一化速度公式和 `upright` 的死区二次罚公式方向都对，只是系数配错了——energy 应该为 0，forward 应该去饱和（给高速更大梯度），stability 应该加强以支撑更快步态。

### selected_intervention
- 移除 `energy_penalty`（`energy_weight: 0.05 → 0`，直接删除该组件）。
- `

下面是环境描述：
本环境是一个 2D 双足 locomotion 任务。智能体需要控制两条腿的髋关节和膝关节，在非平坦地形上持续、稳定地前进，尽可能走得远且快，同时尽量减少能量消耗。核心目标是保持稳定的双足步态并持续向前推进；摔倒会立即终止回合。能量消耗是明确但次级的附属目标，不应与“保持平衡、向前移动”的主目标混淆。

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

# 上一轮训练反馈 (iter_2, score=187.888658)
# Training Feedback

## Final-policy outcome
score=187.888658, len=803.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-40.460825, 322.007954]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_progress_reward | 332.108572 | 89.2% | 89.2% | 100.0% |
| energy_penalty | -34.192022 | -9.2% | 9.2% | 100.0% |
| upright_stability_penalty | -5.878028 | -1.6% | 1.6% | 74.9% |

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
| -96.6 | -99.7 | -94.7 | -84.3 | -81.4 | -47.9 | 186.5 | 192.7 | 193.0 | 228.8 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| energy_penalty | -0.458 | -0.508 | -0.741 | -2.126 | -2.632 | -8.416 | -33.247 | -31.522 | -33.937 | -39.300 |
| forward_progress_reward | 10.003 | 6.462 | 12.327 | 28.310 | 31.647 | 86.022 | 360.954 | 354.123 | 361.336 | 376.813 |
| upright_stability_penalty | -0.017 | -0.123 | -0.000 | -0.956 | -0.296 | -4.326 | -17.343 | -19.879 | -7.821 | -10.911 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| energy_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| forward_progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| upright_stability_penalty | 48.6% | 43.0% | 5.1% | 52.6% | 45.0% | 71.7% | 75.5% | 84.2% | 72.7% | 80.2% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- forward progress (primary) ---
    forward_weight = 1.0
    speed_scale = 1.0
    forward_speed = 0.5 * (obs[2] + next_obs[2])
    forward_progress_reward = (
        forward_weight
        * forward_speed
        / ((forward_speed ** 2 + speed_scale ** 2) ** 0.5)
    )

    # --- upright stability (health constraint) ---
    angle_deadband = 0.5
    angvel_deadband = 1.5
    angle_weight = 1.0
    angvel_weight = 0.05
    eps = 1e-6

    avg_angle = 0.5 * (obs[0] + next_obs[0])
    avg_angvel = 0.5 * (obs[1] + next_obs[1])

    angle_raw = abs(avg_angle) - angle_deadband
    angle_excess = 0.5 * (angle_raw + (angle_raw ** 2 + eps) ** 0.5)
    angle_penalty = angle_weight * angle_excess ** 2

    angvel_raw = abs(avg_angvel) - angvel_deadband
    angvel_excess = 0.5 * (angvel_raw + (angvel_raw ** 2 + eps) ** 0.5)
    angvel_penalty = angvel_weight * angvel_excess ** 2

    upright_stability_penalty = -(angle_penalty + angvel_penalty)

    # --- energy efficiency (secondary, previously missing) ---
    energy_weight = 0.05
    torque_sq = action[0] ** 2 + action[1] ** 2 + action[2] ** 2 + action[3] ** 2
    energy_penalty = -energy_weight * torque_sq

    total_reward = forward_progress_reward + upright_stability_penalty + energy_penalty

    components = {
        "forward_progress_reward": float(forward_progress_reward),
        "upright_stability_penalty": float(upright_stability_penalty),
        "energy_penalty": float(energy_penalty)
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
