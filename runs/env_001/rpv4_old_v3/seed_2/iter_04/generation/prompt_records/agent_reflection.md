# Prompt Record

## System Prompt

```text
你是奖励函数诊断与修订 Agent。先用训练证据解释失败，再选择最小且可验证的干预。你的目标不是匹配某个已知环境或骨架名称，而是改善外部任务表现。

# 证据边界

- 只根据环境事实摘要理解任务、观测和动作，不猜测环境身份，不发明未声明变量。
- feedback来自训练后固定策略的同一批评估轨迹。`episode_sum_mean`表示每回合有符号累计量，`magnitude_share`表示绝对累计量份额，`signed_share`保留净方向，`active_rate`表示非零触发率（旧日志中的`nonzero_rate`）。
- 组件统计是观察证据，不是因果贡献。必须结合score、episode_length、terminated/truncated、历史修改及其结果判断。
- 不同时间语义不可直接比较：逐步差分、持续状态值、惩罚和稀疏事件bonus不能套同一个比例阈值。
- 不得仅因任务描述出现“跳跃、着陆、抓取”等语义，就断言对应状态量是缺失奖励职责。新增职责必须有轨迹行为、终止分布、组件激活或历史干预结果支持；证据不足时明确写“未知”，优先保持best主结构并提出最小可证伪修改。
- episode达到时间上限且失败终止很少时，首先判断现有主信号是否已经实现稳定行为、剩余差距是否来自效率或主目标强度；没有行为证据时，不为动作过程本身添加proxy。

# 唯一决策流程

按顺序完成，不能因知识库或工具命中某个变换就跳级。

## 0. 阅读你的修改自传

在做任何诊断之前，先读 User prompt 的 **#2 你的修改自传**。它记录了你过去每一轮改了什么组件、怎么改的、结果如何。回答三个问题再进入下一步：

1. **你过去改过哪些组件？每个改了几次？** 如果同一个组件已经被改了 4+ 次，你的下一个修改很可能是在重复过去。
2. **哪些修改让分数涨了？哪些让分数跌了？** 如果某个结构（如"近地引导+接触奖励"）曾经带来 +200 分的突破，你后来是否偏离了它？偏离的结果如何？
3. **你是否在同一个决策上来回切换？** 比如加了某机制 → 删掉 → 又加回来 → 又删掉。如果你发现自己在两个方向之间摇摆，说明你需要锁定一个已被证明有效的版本，只在系数层面微调。

读完自传后，如果你发现自己的修改历史有明显的循环或重复模式，在 `evidence` 字段里提及它。

## 1. 行为与历史诊断

User prompt 里的 #4 Subagent 诊断报告已经替你完成了以下工作，从这里出发：

- **骨架演进表** — 并排了同一骨架各轮的 score、len、每个组件的 ep_sum 和 active_rate。纵向扫一眼就能看到："改了 A 那轮，B 塌了，score 跌了"。这是你的核心决策依据。
- **语义演化** — Subagent 已将组件按语义功能分组（接近信号/着陆信号/稳定性信号/完成代理），追踪同一功能在多轮中的名称变化和设计方向切换。如果某功能在"稀疏bonus↔连续奖励"、"状态值↔改善量"之间切换了 ≥ 2 次，Subagent 已在表中标记。**在决定修改某个组件之前，先看语义演化表确认你不是在重复过去的方向切换。**
- **组件健康度** — 每个组件一行判定（健康/死组件/淹没/错位），告诉你哪个信号在驱动 agent、哪个形同虚设。

结合 per-step 量级（`ep_sum ÷ len`），回答两个问题：

1. **agent 学到了什么行为？**
2. **哪个组件最值得干预？** 不一定是变化最大的——如果 A 涨了 B 崩了，解释因果链。

一次只选择一个干预目标。current 明显差于 best 时，以 best 代码为基础做最小修改。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

## 3. 选择干预层级

Subagent 诊断报告会明确告诉你哪个组件最需要干预。**它的推荐是你的起点。** 如果你不同意它的推荐，你必须在 response 里解释为什么不采纳。

先用 `episode_sum_mean ÷ len` 算出每个组件的 **per-step 量级**，对比谁在真正驱动行为。

然后逐组件诊断，每次只改一个组件：

**如果 per-step 量级明显失衡，但数学形态合理** → 调系数。强的组件在产生健康行为（score 涨、terminated 涨）就保留它、提弱组件；强的组件在驱动病态行为就削弱它。

**如果某个组件数学形态本身传达了错误激励** → 改这个组件的公式形式。典型信号：值在训练全程几乎不变（死组件）、乘积塌缩导致 active_rate 接近零、持续状态值无界累积导致 agent 赖着不走、全局惩罚全覆盖压制探索。参考变换方向：

| 证据 | 变换 |
|---|---|
| 事件几乎不触发 | 稀疏 → 连续 |
| 极端值支配 | 无界 → 有界 |
| 占据状态即获奖 | 状态值 → 改善量 |
| 乘积塌缩 | 乘积 → 加性/几何平均 |
| proxy 与外部分数脱钩 | 代理 → 任务完成对齐 |

**如果多轮迭代从未拿到可接受分数，且 Subagent 语义演化表显示同一功能已被修改 3+ 次仍未突破，或某语义功能在相反设计方向之间切换 ≥ 2 次** → 换主信号框架。选这一步前必须确认：历史上是否已有迭代拿到过不错的分数？如果有——你不是需要换框架，你需要在 best 基础上做上述两种修改之一。

# 代码约束

- 禁止terminal_success_reward、terminal_failure_penalty、original_reward。
- 只能使用环境事实摘要声明的obs、next_obs、action和info字段，不得发明字段、切片维度或新输入。
- 第一个Python code block只能包含一个完整的`compute_reward`函数；不要写import、class、try/except或额外函数，不要使用self。
- 禁止eval/exec/open，禁止使用original_reward或原始环境reward。
- 需要平方根时使用`** 0.5`，禁止import numpy。需要指数形式时使用`2.718281828 ** exponent`，或改用`1/(1+k*x)`、`max(0,1-x/D)`等无需库的bounded表达式。
- 除Level 3重建外，每轮只修改一个目标组件，不顺带调整其他组件。
- 函数签名必须是：`def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):`
- 返回`(float(total_reward), components)`；components只放总公式中直接出现的奖励组件，不放total_reward和中间调制器。

# 输出

先用以下固定字段各写一句，不复述输入表格：

1. `evidence`：支持判断的外部结果、组件证据和上一轮结果；
2. `behavior_diagnosis`：策略当前的失败行为；
3. `signal_completeness`：必要职责是否完备、可达；
4. `selected_level`：Level 1、Level 2或Level 3及触发条件；
5. `selected_intervention`：唯一目标组件及具体修改；
6. `falsifiable_hypothesis`：为什么该修改应改善策略；
7. `expected_next_round`：下一轮哪些指标应如何变化；
8. `main_risk`：最可能引入的新漏洞。

然后立即输出完整Python代码。预期必须能被下一轮反馈证伪。
```

## User Prompt

```markdown
# 1. Search objective
- target_score: 200.000000
- current_score: -203.446380
- gap_to_target: 403.446380
- target_achievement_ratio: -101.723%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `landing_velocity_penalty` → 分数 -156.0 → -119.5（+36.5）✅ 大幅改善
- **你做了什么**：将`landing_velocity_penalty`组件整体替换为`success_landing_bonus`，基于next_obs的接触、位置、速度、角度给出稀疏大正奖励。
- **你为什么这么做**：添加成功奖励后，策略在成功沾盘瞬间获得+100，远大于所有单步惩罚，应能强化接近并软着陆的行为，使最终得分由负转正、terminated中成功终止占比增加。
- **回合长度**：len: 154→68

### v3：修改 `crash_penalty` → 分数 -119.5 → -203.4（-83.9）❌ 明显退步
- **你做了什么**：新增 `crash_penalty` 组件，在接触地面但未满足软着陆条件、或 x 位置超出视区边界时给予 -100，与 success_landing_bonus 互斥，其余组件不变。
- **你为什么这么做**：添加与成功对称的 crash 惩罚后，agent 将被迫避免高速接触和越界，从而逐渐探索减速并在靠近目标时精准控制，最终触发 success_landing_bonus，score 应上升，终止方式可能出现成功终止或 len 增加。
- **回合长度**：len: 68→154

## 你的修改轨迹（按组件分组）

- **`landing_velocity_penalty`**：修改了 **1 次**，分数变化：v2(+36)
- **`crash_penalty`**：修改了 **1 次**，分数变化：v3(-84)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: -203.446380）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 距离势能塑造 (progress_shaping)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 坠毁/越界惩罚 (crash_penalty)
    out_of_bounds = abs(next_x_pos) > 1.5
    # 接触地面但未满足软着陆条件，或出界，均视为失败
    crash = (not landed) and (contact_any or out_of_bounds)
    crash_penalty_val = -100.0 if crash else 0.0

    # 4. 姿态惩罚
    angle_penalty_val = -0.5 * (body_angle ** 2)

    # 5. 动作代价
    action_cost_val = -0.02 if action != 0 else 0.0

    # 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        crash_penalty_val +
        angle_penalty_val +
        action_cost_val
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "crash_penalty": crash_penalty_val,
        "angle_penalty": angle_penalty_val,
        "action_cost": action_cost_val
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告 — iter_03（本轮）

## 1. 行为事实
- **得分**：score=-203.45，range=[-270.27, -158.78]
- **终止方式**：全部因接触地面或越界而提前终止（terminated=20/20）；无时间上限截断（truncated=0）。
- **回合长度**：len=154.15
- **Checkpoint 轨迹**：40% 步数时得分 0.00，80% 步数时得分 0.00（均无提升）。
- **奖励组成**：
  - 正向贡献仅有 `progress_shaping` `+1.49`（sign=+17.9%，mag=52.6%）。
  - 惩罚端主要由 `angle_penalty`（-2.46）和 `action_cost`（-1.47）构成。
  - `crash_penalty` 与 `success_landing_bonus` 均完全未触发（active_rate=0.0%，ep_sum=0.0）。

## 2. 骨架演进表

| iter | score | len | term | progress_shaping_ep_sum | progress_active | success_landing_bonus_ep_sum | success_active | landing_velocity_penalty_ep_sum | lvp_active | crash_penalty_ep_sum | crash_active | angle_penalty_ep_sum | angle_active | action_cost_ep_sum | action_active |
|------|--------|------|------|--------------------------|-----------------|------------------------------|----------------|---------------------------------|-------------|----------------------|--------------|----------------------|--------------|--------------------|---------------|
| 1 | -155.98 | 154.40 | 20/0 | 1.58 | 100.0% | — | — | -0.07 | 5.3% | — | — | -1.14 | 100.0% | -1.80 | 58.4% |
| 2 | -119.51 | 68.30 | 20/0 | 5.58 | 100.0% | 5.00 | 0.1% | — | — | — | — | -0.20 | 100.0% | -0.07 | 5.2% |
| 3 | -203.45 | 154.15 | 20/0 | 1.49 | 100.0% | 0.00 | 0.0% | — | — | 0.00 | 0.0% | -2.46 | 100.0% | -1.47 | 47.7% |

*ter: terminated/truncated。active = active_rate。*

## 3. 数字标记

- **全覆盖惩罚**：`angle_penalty` active_rate=100.0% 且 signed_share=-29.7%，属于持续激活的纯惩罚项。
- **死组件**：`crash_penalty` active_rate=0.0%，`success_landing_bonus` active_rate=0.0%。两者在 iter_03 中完全未生效。
- **量级对比**（每步均值 ep_sum÷len）：
  - 最大正贡献：`progress_shaping` = +0.00965
  - 最大负贡献：`angle_penalty` = -0.01596
  - 最小两个：`crash_penalty` = 0.0，`success_landing_bonus` = 0.0
- **修改变化（iter_02 → iter_03 Δ）**：
  | 组件 | Δ ep_sum | Δ active_rate |
  |------|----------|---------------|
  | progress_shaping | -4.10 | 0.0pp |
  | success_landing_bonus | -5.00 | -0.1pp |
  | angle_penalty | -2.26 | 0.0pp |
  | action_cost | -1.40 | +42.5pp |
  | crash_penalty（新增） | +0.00 | +0.0% |

## 4. 语义演化（跨轮组件功能追踪）

| 语义功能 | 各轮组件名 | 各轮 ep_sum_mean | 方向切换 |
|---------|----------|-----------------|---------|
| 接近/进度信号 | iter_1: progress_shaping → iter_2: progress_shaping → iter_3: progress_shaping | 1.58 → 5.58 → 1.49 | 0 次（始终为连续 Δ 势能） |
| 着陆/接触信号 | iter_1: landing_velocity_penalty（死） → iter_2: success_landing_bonus（0.1%） → iter_3: success_landing_bonus（死） + crash_penalty（死） | -0.07 → 5.00 → 0.00+0.00 | 1 次：连续惩罚（v1）→ 稀疏 bonus（v2→v3） |
| 稳定性/约束信号 | iter_1: angle_penalty → iter_2: angle_penalty → iter_3: angle_penalty | -1.14 → -0.20 → -2.46 | 0 次（始终为二次型姿控） |
| 完成/成功代理 | 无 | — | — |

**方向切换说明**：着陆/接触信号在 v1 使用连续速度惩罚（激活率仅 5.3%），v2 切换为稀疏大 bonus（激活率 0.1%），v3 新增对称稀疏 crash 惩罚但激活率为 0%，同一功能角色的信号从“连续惩罚”切换到“稀疏 bonus+稀疏惩罚”，发生 1 次方向切换。

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Lunar Lander 风格任务奖励函数 v2。

    修改:
    - 移除无效的 landing_velocity_penalty，新增 success_landing_bonus:
      当 next_obs 满足软着陆条件时给予 +100 稀疏大奖励，解决 v1 无成功信号的问题。
    - 其余组件保持不变。
    """
    # 0. 解析观测
    x_pos, y_pos = obs[0], obs[1]
    body_angle = obs[4]

    next_x_pos, next_y_pos = next_obs[0], next_obs[1]
    next_x_vel, next_y_vel = next_obs[2], next_obs[3]
    next_body_angle = next_obs[4]
    next_left_contact, next_right_contact = next_obs[6], next_obs[7]

    # 1. 主学习信号: 距离势能塑造 (goal_progress_guidance)
    dist = (x_pos ** 2 + y_pos ** 2) ** 0.5
    next_dist = (next_x_pos ** 2 + next_y_pos ** 2) ** 0.5
    potential_curr = -dist
    potential_next = -next_dist
    progress_reward = 5.0 * (potential_next - potential_curr)

    # 2. 成功着陆奖励 (success_landing_bonus)
    # 判定条件: 距离目标极近、速度很小、姿态竖直、至少一个支撑脚触地
    contact_any = (next_left_contact > 0.5) or (next_right_contact > 0.5)
    landed = (
        next_dist < 0.1 and
        abs(next_x_vel) < 0.2 and
        abs(next_y_vel) < 0.2 and
        abs(next_body_angle) < 0.1 and
        contact_any
    )
    success_bonus = 100.0 if landed else 0.0

    # 3. 姿态惩罚
    angle_penalty = -0.5 * (body_angle ** 2)

    # 4. 动作代价
    action_cost = -0.02 if action != 0 else 0.0

    # 5. 总奖励
    total_reward = (
        progress_reward +
        success_bonus +
        angle_penalty +
        action_cost
    )

    components = {
        "progress_shaping": progress_reward,
        "success_landing_bonus": success_bonus,
        "angle_penalty": angle_penalty,
        "action_cost": action_cost
    }

    return float(total_reward), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| action_cost | -0.006 | -0.015 | -0.009 | 28.0% | 74.6% | +46.6pp |
| angle_penalty | -0.004 | -0.034 | -0.030 | 100.0% | 100.0% | +0.0pp |
| crash_penalty | — (新增) | -0.425 | 新增 | — | 0.4% | — |
| progress_shaping | 0.080 | -0.007 | -0.087 | 100.0% | 100.0% | +0.0pp |
| success_landing_bonus | 0.208 | 0.002 | -0.207 | 0.2% | 0.0% | -0.2pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -119.51 | -203.45 | -83.94 |
| 平均回合长度 | 68.3 | 154.2 | +85.9 |
| 终止回合数 | 20/20 | 20/20 | +0 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-203.446380, len=154.150000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-270.266370, -158.776888]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_shaping | 1.487613 | 17.9% | 52.6% | 100.0% |
| angle_penalty | -2.459905 | -29.7% | 29.7% | 100.0% |
| action_cost | -1.472000 | -17.8% | 17.8% | 47.7% |
| crash_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |
| success_landing_bonus | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 11/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 的飞行器/着陆器轨迹优化任务。智能体从一个靠近顶部中央的位置出发，带有随机初始作用力。  
**主目标**：尽快到达画面中央的目标平台，并安全停靠（soft landing），即要求位置收敛到平台上、速度很低、姿态竖直、支撑脚安全触地。  
**次目标**：在完成任务的前提下，尽量少使用引擎推力（节省燃料），同时保持姿态稳定。  
**不应混淆的目标**：该任务不是纯生存或平衡任务——到达目标平台是刚性的成功条件，能量节省只是辅助约束，不是多目标的平等折衷。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: [8]
- **dtype**: float32 (推断)
- **obs[0]**: `x_position`，水平坐标（相对于目标平台中心），用于衡量横向偏移，奖励可用 ✅
- **obs[1]**: `y_position`，垂直坐标（相对于平台高度），用于衡量高度/下降进程，奖励可用 ✅
- **obs[2]**: `x_velocity`，水平线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[3]**: `y_velocity`，垂直线速度，用于减速奖励/惩罚，奖励可用 ✅
- **obs[4]**: `body_angle`，机体角度（可能以弧度为单位，0 表示竖直），奖励可用 ✅
- **obs[5]**: `angular_velocity`，角速度，用于稳定姿态，奖励可用 ✅
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（0/1），指示是否触地，奖励可用 ✅

## 4. 动作空间 action_space
- **type**: Discrete
- **n**: 4
- **动作含义**：
  - `0`: `no_engine` – 不激活任何引擎，仅靠惯性飞行。
  - `1`: `left_orientation_engine` – 点火左定向引擎，产生一个使机体逆时针（或某一方向）旋转的力矩。
  - `2`: `main_engine` – 点火主引擎，产生向上的推力（对抗重力并控制下降速度）。
  - `3`: `right_orientation_engine` – 点火右定向引擎，产生与动作 1 相反的旋转力矩。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `MASKED_STEP_SOURCE` 的 `terminated` 计算逻辑：
- **crash_or_body_contact**：躯体发生碰撞或身体支撑点接触地面。此情景可能包括：
  - 成功的软着陆（接触时速度极低、姿态正）。
  - 失败的高速撞击/翻倒等。
- **horizontal_position_outside_viewport**：x 坐标超出视区边界，明确失败。
- **body_not_awake_or_settled**：躯体进入睡眠状态或已稳定（settled）。这可能是：
  - 成功着陆后自然睡眠。
  - 坠毁后物理引擎判定身体安定，属于失败。

**总结终止模式**：
- **success-like termination**：`body_not_awake_or_settled` 在速度接近于 0、位置接近目标平台、接触标志为 1 且角度接近 0 时发生。
- **failure-like termination**：`horizontal_position_outside_viewport`；或因高速、大角度导致的 `crash_or_body_contact`/`body_not_awake_or_settled`。
- **ambiguous termination**：`crash_or_body_contact`/`body_not_awake_or_settled` 需结合 next_obs 判定成败。
- **truncation**：未提及，但有最大 episode 长度（通常存在），应在 reward 中注意时间步惩罚。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无任何显式标志）
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因为没有提供）

> 因此，任何成功/失败的判断必须从 `next_obs` 与 `terminated` 的组合中推断，而不能依赖 `info`。

## 7. 可用于奖励函数的信号
- **position**：`x_position`, `y_position`（相对目标），可计算到目标的欧氏距离或分别控制。
- **velocity**：`x_velocity`, `y_velocity`，可设计近地减速、稳定约束。
- **orientation**：`body_angle`，鼓励竖直。
- **angular**：`angular_velocity`，鼓励旋转抑制。
- **contact**：`left_support_contact`, `right_support_contact`，标志着陆接触已发生，可配合速度用于判定软着陆。
- **action/engine**：`action` 选择，可用于惩罚引擎使用（燃料节省）。
- **other**：`terminated` 标志（作为环境返回，但 `compute_reward` 接口未直接提供，可通过在调用 `compute_reward` 时传入 `done` 标志获取；我们可以在训练循环中以额外参数传递，但接口中只有 `next_obs` 和 `info`；`terminated` 通常可在调用时通过另一个变量获得，若必须在 `compute_reward` 中获取，需要从外部传入——在此假设可以在调用时提供 `done` 标志，否则可通过 `next_obs` 的特征推断终止状态。由于题干未明确禁止，但谨慎起见，直接使用 `done` 需要额外参数，在标准接口中不鼓励，所以这里只强调可从 `next_obs` 和 `info` 推断。但 info 空，所以成功稀疏奖励只能基于 `next_obs` 的状态是否符合成功条件给出，而不依赖外部 `done`。所以我们仍可以将“是否处于终止且状态符合成功条件”作为奖励计算的一部分，但必须在训练循环的每个 step 结束时调用，需要能够判断 `done`，这部分由训练框架负责。我们此处声明信号可用，但不具体到实现细节。）

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_cost + angle_penalty + landing_velocity_penalty + progress_shaping | -155.98 | -155.98 | 0.00 | 154.40 | action_cost=-0.013 angle_penalty=-0.015 landing_velocity_penalty=-0.118 progress_shaping=0.035 | new_best |
| 2 | action_cost + angle_penalty + progress_shaping + success_landing_bonus | -119.51 | -119.51 | 0.00 | 68.30 | action_cost=-0.006 angle_penalty=-0.004 progress_shaping=0.080 success_landing_bonus=0.208 | new_best |
| 3 | action_cost + angle_penalty + crash_penalty + progress_shaping + success_landing_bonus | -203.45 | -119.51 | -83.94 | 154.15 | action_cost=-0.015 angle_penalty=-0.034 crash_penalty=-0.425 progress_shaping=-0.007 success_landing_bonus=0.002 | no_meaningful_improvement |

```
