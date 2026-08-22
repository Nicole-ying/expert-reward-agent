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

User prompt 里的 #4 Subagent 诊断报告已经替你完成了三件事，从这里出发：

- **骨架演进表** — 并排了同一骨架各轮的 score、len、每个组件的 ep_sum 和 active_rate。纵向扫一眼就能看到："改了 A 那轮，B 塌了，score 跌了"。这是你的核心决策依据。
- **组件健康度** — 每个组件一行判定（健康/死组件/淹没/错位），告诉你哪个信号在驱动 agent、哪个形同虚设。
- **干预建议** — "最需要干预的组件是 X，因为 per-step 量级只有 Y 的 1/50"。

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

**如果多轮迭代从未拿到可接受分数，且 Subagent 诊断和 Memory 表确认当前骨架方向错误** → 换主信号框架。选这一步前必须确认：历史上是否已有迭代拿到过不错的分数？如果有——你不是需要换框架，你需要在 best 基础上做上述两种修改之一。

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
- current_score: 212.944108
- gap_to_target: -12.944108
- target_achievement_ratio: 106.472%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `soft_landing_penalty` → 分数 -40.0 → 217.2（+257.2）✅ 大幅改善
- **你做了什么**：唯一目标组件为soft_landing_penalty，将其替换为landing_success_reward，条件为双腿接触、近中心、低速、小姿态角，持续给予正奖励。
- **你为什么这么做**：添加着陆成功信号后，agent会在dist接近0时获得持续激励，从而学会减速并维持姿态，减少失败硬着陆。
- **回合长度**：len: 154→484

### v3：修改 `crash_prevention_y` → 分数 217.2 → 212.9（-4.3）➖ 小幅退步
- **你做了什么**：删除 penalty_crash_x 与 penalty_crash_y 的计算和 components 条目，保持其余组件（goal_proximity、landing_success_reward、living_penalty）完全不变...
- **你为什么这么做**：删除这些零贡献组件后，奖励总和与各组件 ep_sum 在统计上应与本轮完全相同，得分、回合长度、终止分布均不发生变化。
- **回合长度**：len: 484→512

## 你的修改轨迹（按组件分组）

- **`soft_landing_penalty`**：修改了 **1 次**，分数变化：v2(+257)
- **`crash_prevention_y`**：修改了 **1 次**，分数变化：v3(-4)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: 212.944108）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15
    landing_ang_thresh = 0.1
    w_landing = 1.0

    contacts = left_contact + right_contact
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 时间惩罚
    living_penalty = -0.002

    total_reward = reward_prox + reward_landing + living_penalty

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告

## 1. 行为事实
- 得分：score = 212.94，range = [69.66, 274.05]
- 终止方式：terminated 16/20（成功着陆），truncated 4/20（超时）
- 回合长度：len = 511.90
- Checkpoint 轨迹（官方得分）：
  - 40% 步数：score_mean = 0.00
  - 80% 步数：score_mean = 0.00
  → 从 40% 到 80% 官方得分均为 0，持平

## 2. 骨架演进表

| iter | score | len | term | goal_proximity_ep_sum | goal_proximity_active | landing_success_reward_ep_sum | landing_success_reward_active | living_penalty_ep_sum | living_penalty_active | soft_landing_penalty_ep_sum | soft_landing_penalty_active | crash_prevention_x_ep_sum | crash_prevention_x_active | crash_prevention_y_ep_sum | crash_prevention_y_active |
|------|--------|--------|---------------------------------|------------------------|------------------------|-------------------------------|-------------------------------|------------------------|------------------------|-----------------------------|-----------------------------|----------------------------|----------------------------|----------------------------|----------------------------|
| 1    | -40.04 | 154.05 | terminated 20/20, truncated 0   | 2.1388                 | 100.0%                 | –                             | –                             | -0.3081                | 100.0%                 | -0.0526                     | 1.5%                        | 0.0                        | 0.0%                       | 0.0                        | 0.0%                       |
| 2    | 217.21 | 484.10 | terminated 18/20, truncated 2   | 2.5678                 | 97.6%                  | 36.45                         | 7.5%                          | -0.9682                | 100.0%                 | –                           | –                           | 0.0                        | 0.0%                       | 0.0                        | 0.0%                       |
| 3    | 212.94 | 511.90 | terminated 16/20, truncated 4   | 2.6382                 | 97.6%                  | 37.25                         | 7.3%                          | -1.0238                | 100.0%                 | –                           | –                           | –                          | –                          | –                          | –                          |

*注：iter2 的 crash_prevention_x / y 仍保留在代码中，但每回合均值为 0，active_rate 为 0；iter3 代码中已移除这两个组件，标记为“–”。*

## 3. 数字标记

- **主导**：landing_success_reward 的 magnitude_share = 90.7% > 80%
- **死组件**：本轮（iter3）无活跃组件 active_rate < 2%
- **淹没**：living_penalty 的 active_rate = 100.0% > 90%，magnitude_share = 2.5% < 3%
- **全覆盖惩罚**：living_penalty 的 active_rate = 100.0% > 95% 且 signed_share = -2.5%（负）
- **量级对比**（ep_sum ÷ len，按本组数值）：
  - 最大的两个：
    1. landing_success_reward: 37.25 / 511.9 ≈ 0.0728
    2. goal_proximity: 2.6382 / 511.9 ≈ 0.00515
  - 最小的两个：
    1. living_penalty: -1.0238 / 511.9 ≈ -0.00200
    2. goal_proximity: ≈ 0.00515（与上面并列第二；因仅有三个组件，故列出该值）
- **修改变化**（iter3 vs iter2，公共组件）：
  - landing_success_reward: ep_sum Δ = +0.80，active_rate Δ = -0.2 个百分点
  - goal_proximity: ep_sum Δ = +0.0704，active_rate Δ = 0.0 个百分点
  - living_penalty: ep_sum Δ = -0.0556，active_rate Δ = 0.0 个百分点

# 4.5. 历史最佳奖励函数代码
如果 Subagent 诊断建议回退到某个历史最佳版本，或你判断当前方向错误，应基于此最佳代码做最小修改，而不是从当前失败代码出发或全盘重建。
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 提取观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    body_angle = next_obs[4]
    # 腿接触信息（0/1）
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 距离计算
    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # 1. 主学习信号：goal_proximity（距离改善差分）
    w_prox = 2.0
    reward_prox = w_prox * (dist - next_dist)

    # 2. 着陆成功奖励（替代原 soft_landing_penalty）
    # 当双腿接触、位置接近原点、速度小、姿态水平时给予持续正奖励
    landing_dist_thresh = 0.2
    landing_vel_thresh = 0.15   # 用于水平/垂直线速度绝对值
    landing_ang_thresh = 0.1    # 角度绝对值
    w_landing = 1.0

    contacts = left_contact + right_contact           # 0, 1, 2
    near_center = next_dist < landing_dist_thresh
    low_speed = abs(next_vx) < landing_vel_thresh and abs(next_vy) < landing_vel_thresh
    upright = abs(body_angle) < landing_ang_thresh

    # 只有两条腿都与地面接触且满足其他条件才算成功着陆
    landing_success = (contacts == 2) and near_center and low_speed and upright
    reward_landing = w_landing if landing_success else 0.0

    # 3. 碰撞预防（保留原有逻辑，尝试微调使之可能触发）
    x_bound = 0.9
    w_crash_x = 10.0
    penalty_crash_x = 0.0
    if abs(next_x) > x_bound:
        penalty_crash_x = -w_crash_x * (abs(next_x) - x_bound) ** 2

    danger_vy = -0.5
    w_crash_y = 5.0
    penalty_crash_y = 0.0
    if next_y < -0.1 and next_vy < danger_vy:
        penalty_crash_y = -w_crash_y * (next_vy - danger_vy) ** 2

    # 4. 时间惩罚
    living_penalty = -0.002

    total_reward = (reward_prox + reward_landing +
                    penalty_crash_x + penalty_crash_y +
                    living_penalty)

    components = {
        "goal_proximity": reward_prox,
        "landing_success_reward": reward_landing,
        "crash_prevention_x": penalty_crash_x,
        "crash_prevention_y": penalty_crash_y,
        "living_penalty": living_penalty
    }

    return float(total_reward), components
```

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| crash_prevention_x | -0.000 | — (新增) | 已删除 | 0.0% | — | — |
| crash_prevention_y | -0.001 | — (新增) | 已删除 | 0.0% | — | — |
| goal_proximity | 0.013 | 0.012 | -0.001 | 99.8% | 99.6% | -0.2pp |
| landing_success_reward | 0.312 | 0.348 | +0.036 | 31.2% | 34.8% | +3.6pp |
| living_penalty | -0.002 | -0.002 | 0.000 | 100.0% | 100.0% | 0.0pp |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | 217.21 | 212.94 | -4.27 |
| 平均回合长度 | 484.1 | 511.9 | +27.8 |
| 终止回合数 | 18/20 | 16/20 | -2 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=212.944108, len=511.900000, terminated=16/20, truncated=4/20, reward_errors=0
score_range=[69.659258, 274.047242]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_reward | 37.250000 | 90.7% | 90.7% | 7.3% |
| goal_proximity | 2.638238 | 6.4% | 6.8% | 97.6% |
| living_penalty | -1.023800 | -2.5% | 2.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：从画面顶部中央附近的随机初态出发，控制二维飞行器尽快到达中心目标垫并稳定着陆（速度接近零、姿态接近水平、支撑腿接触）。  
次目标：在完成主目标的前提下，尽可能少用引擎推力（节能）。  
混淆澄清：该任务不是纯粹的速度竞赛，也不是单纯的节能悬停，核心是实现“成功、快速、节能”的精确着陆。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `[8]`
- dtype: `float32`（推断）
- obs[0]: `x_position`，水平坐标（相对目标垫），可用于奖励接近目标，reward_usable: true
- obs[1]: `y_position`，垂直坐标（相对垫高），可用于奖励接近目标和防越界，reward_usable: true
- obs[2]: `x_velocity`，水平线速度，可用于奖励稳定着陆和禁止硬着陆，reward_usable: true
- obs[3]: `y_velocity`，垂直线速度，同上，reward_usable: true
- obs[4]: `body_angle`，机体俯仰角，可用于奖励姿态稳定，reward_usable: true
- obs[5]: `angular_velocity`，角速度，可用于奖励姿态平滑，reward_usable: true
- obs[6]: `left_support_contact`，左支撑腿接触标志（0/1），用于判断着陆成功，reward_usable: true
- obs[7]: `right_support_contact`，右支撑腿接触标志，同上，reward_usable: true

## 4. 动作空间 action_space
- type: `Discrete`
- n: 4
- action 0: `no_engine`，不做任何推力/旋转，完全靠惯性
- action 1: `left_orientation_engine`，激发侧向方向引擎（改变机体转角，可能逆时针）
- action 2: `main_engine`，激发主引擎（沿机体朝向产生推力）
- action 3: `right_orientation_engine`，激发另一侧方向引擎（改变机体转角，可能顺时针）

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（当飞行器停稳、身体不再移动时触发，可能在正确着陆后发生）
- failure-like termination:  
  - `crash_or_body_contact`（主体非支撑部分碰地或坠毁）  
  - `horizontal_position_outside_viewport`（水平方向飞出视野）
- ambiguous termination: `body_not_awake_or_settled`在未到达目标垫时也可能触发（如坠落后停在边界外），但这时的终止不应视为成功；需要通过观察是否在目标垫上和接触确认。
- truncation: 代码中 `truncated` 恒为 `False`，无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}`（空字典，无任何 info 字段）
- forbidden_or_uncertain_info_fields: 不存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]`，任何推断成功/失败的逻辑必须从 `obs` / `next_obs` 的状态条件进行判定。

## 7. 可用于奖励函数的信号
- **位置**：`x_position`(obs[0]), `y_position`(obs[1]) — 水平与垂直相对目标垫，可直接用于奖励接近和判断边界。
- **速度**：`x_velocity`(obs[2]), `y_velocity`(obs[3]) — 线速度，用于奖励减速、软着陆。
- **姿态**：`body_angle`(obs[4]) — 角度（可能是弧度或归一化角），用于奖励水平姿态；`angular_velocity`(obs[5]) — 用于奖励姿态平稳。
- **接触**：`left_support_contact`(obs[6]), `right_support_contact`(obs[7]) — 布尔值，标志支撑腿是否接触地面，可奖励着陆成功。
- **动作/引擎**：`action` — 离散索引，可对主引擎和方向引擎使用进行惩罚，实现节能。

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_prevention_x + crash_prevention_y + goal_proximity + living_penalty + soft_landing_penalty | -40.04 | -40.04 | 0.00 | 154.05 | crash_prevention_x=-0.000 crash_prevention_y=-0.007 goal_proximity=0.021 living_penalty=-0.002 soft_landing_penalty=-0.012 | new_best |
| 2 | crash_prevention_x + crash_prevention_y + goal_proximity + landing_success_reward + living_penalty | 217.21 | 217.21 | 0.00 | 484.10 | crash_prevention_x=-0.000 crash_prevention_y=-0.001 goal_proximity=0.013 landing_success_reward=0.312 living_penalty=-0.002 | target_solved_new_best |
| 3 | goal_proximity + landing_success_reward + living_penalty | 212.94 | 217.21 | -4.27 | 511.90 | goal_proximity=0.012 landing_success_reward=0.348 living_penalty=-0.002 | target_solved_no_improvement |

```
