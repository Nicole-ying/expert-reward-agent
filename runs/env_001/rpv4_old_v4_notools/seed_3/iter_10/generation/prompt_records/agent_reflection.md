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
错误信息：Reward v10 failed validation: 缺少准确函数签名; 没有发现 components/reward_components/reward_terms 字典赋值; warnings: 建议返回 (float(total_reward), components)。当前 wrapper 兼容 float，但 tuple 返回更利于诊断。; 未发现明显的 total_reward/reward 变量名 (record: runs/env_001/rpv4_old_v4_notools/seed_3/iter_10/generation/validations/reward_v10.validation.json)
这是代码格式修复，不要重新诊断、不要调用工具、不要改变原定修改方向。直接输出修复后的完整 Python 代码。

# 被截断或无效的上一版草稿
```json
{
  "tool": "read_memory"
}
```

下面是环境描述：
本环境是一个 2D 飞行器轨迹优化问题。主体从画面顶部中央附近起始，并带有随机初始力。核心任务是尽快到达并稳定停靠在位于地面中央的目标平台上，同时尽可能减少引擎推力消耗。智能体需要学习自主接近目标、降低速度、保持稳定姿态，并实现安全接触（即两侧支撑腿均接触平台）。次要目标包括：路径平滑、姿态稳定、节约燃料。不可混淆的目标是仅追求最快到达而忽视安全着陆（比如高速撞击）。

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

# 上一轮训练反馈 (iter_9, score=-119.090536)
# Training Feedback

## Final-policy outcome
score=-119.090536, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-141.274164, -95.259451]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 55.000000 | 64.9% | 64.9% | 0.8% |
| speed_penalty | -27.315364 | -32.2% | 32.2% | 100.0% |
| progress_reward | 1.126019 | 1.3% | 1.4% | 100.0% |
| attitude_penalty | -1.093118 | -1.3% | 1.3% | 100.0% |
| contact_reward | 0.220000 | 0.3% | 0.3% | 1.6% |
| energy_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

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
| -121.1 | -122.2 | -117.2 | -114.3 | -115.2 | -92.2 | -120.4 | -118.7 | -123.2 | -119.6 |

## Per-component ep_sum_mean trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | -1.683 | -1.339 | -0.877 | -0.752 | -0.900 | -1.028 | -1.006 | -0.868 | -1.214 | -0.979 |
| contact_reward | 0.240 | 0.250 | 0.240 | 0.250 | 0.240 | 0.280 | 0.230 | 0.240 | 0.220 | 0.250 |
| energy_penalty | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| landing_bonus | 35.000 | 55.000 | 30.000 | 50.000 | 30.000 | 50.000 | 40.000 | 50.000 | 40.000 | 55.000 |
| progress_reward | 1.126 | 1.123 | 1.121 | 1.121 | 1.120 | 1.124 | 1.123 | 1.122 | 1.123 | 1.125 |
| speed_penalty | -27.256 | -27.368 | -27.482 | -27.382 | -27.494 | -27.442 | -27.392 | -27.413 | -27.373 | -27.304 |

## Per-component active_rate trend
| component | cp10% | cp20% | cp30% | cp40% | cp50% | cp60% | cp70% | cp80% | cp90% | cp100% |
|---|---||---||---||---||---||---||---||---||---||---|
| attitude_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| contact_reward | 1.8% | 1.8% | 1.8% | 1.8% | 1.8% | 2.0% | 1.7% | 1.8% | 1.6% | 1.8% |
| energy_penalty | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| landing_bonus | 0.5% | 0.8% | 0.4% | 0.7% | 0.4% | 0.7% | 0.6% | 0.7% | 0.6% | 0.8% |
| progress_reward | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| speed_penalty | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |

# 上一轮奖励函数代码
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    新主信号：基于距离差值的进展奖励 + 速度/姿态约束 + 阶段门控的着陆大奖
    目标：克服自杀式坠毁，引导 agent 学习接近、减速、稳定着陆
    """
    # ========== 权重 ==========
    w_progress    = 1.0    # 进展信号权重（每步靠近的奖励）
    w_speed       = 0.3    # 速度惩罚系数
    w_attitude    = 0.1    # 姿态/角速度惩罚系数
    w_contact     = 0.2    # 双腿接触即时奖励
    w_landing     = 100.0  # 成功着陆大奖

    # ========== 状态提取 ==========
    # 当前步绝对位置
    x, y = obs[0], obs[1]
    # 下一步
    nx, ny   = next_obs[0], next_obs[1]
    vx, vy   = next_obs[2], next_obs[3]
    angle    = next_obs[4]
    ang_vel  = next_obs[5]
    left_con = next_obs[6]
    right_con = next_obs[7]

    # 当前与下一步距离
    current_dist = (x**2 + y**2) ** 0.5
    next_dist    = (nx**2 + ny**2) ** 0.5

    # ========== 组件计算 ==========

    # 1. 稠密进展：每步靠近目标的正奖励（无门控）
    delta_dist = current_dist - next_dist
    progress_reward = w_progress * delta_dist

    # 2. 速度惩罚：抑制高速移动
    abs_speed = abs(vx) + abs(vy)
    speed_penalty = -w_speed * abs_speed

    # 3. 姿态/角速度惩罚（全局但权重适度）
    attitude_penalty = -w_attitude * (abs(angle) + abs(ang_vel))

    # 4. 双腿接触即时反馈
    both_legs = left_con * right_con  # 0 或 1
    contact_reward = w_contact * both_legs

    # 5. 成功着陆大奖（阶段门控：接触、低速、近心）
    landing_condition = (both_legs > 0.5) and (abs_speed < 0.5) and (next_dist < 1.0)
    landing_bonus = w_landing if landing_condition else 0.0

    # 6. 主引擎燃料惩罚（保留但默认权重为 0，待后续观察再启用）
    main_engine_used = 1.0 if action == 2 else 0.0
    energy_penalty = 0.0 * main_engine_used   # 当前关闭

    # ========== 总奖励 ==========
    total_reward = (
        progress_reward
        + speed_penalty
        + attitude_penalty
        + contact_reward
        + landing_bonus
        + energy_penalty
    )

    components = {
        "progress_reward":   progress_reward,
        "speed_penalty":     speed_penalty,
        "attitude_penalty":  attitude_penalty,
        "contact_reward":    contact_reward,
        "landing_bonus":     landing_bonus,
        "energy_penalty":    energy_penalty,
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
