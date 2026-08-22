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
- current_score: 255.276942
- gap_to_target: -55.276942
- target_achievement_ratio: 127.638%

# 2. 你的修改自传 — 你过去每一轮做了什么，结果如何

逐轮阅读，关注你**自己的修改决策和它们的结果**。注意你是否在重复过去的操作。

### v2：修改 `soft_landing` → 分数 -214.2 → -109.6（+104.6）✅ 大幅改善
- **你做了什么**：唯一目标组件为 soft_landing，将其从“低空门控二次惩罚”改为“全高度线性惩罚”，同时纳入姿态角度和角速度，使其在全程产生负向激励，引导 agent 学习减速与稳定姿态。
- **你为什么这么做**：加入全程速度/姿态惩罚后，agent 将学会使用主引擎抵消下沉、利用姿态引擎抑制翻滚，从而降低惩罚，并与 goal_approach 共同引导向中心垫缓慢降落，最终提高 score 并可能产生腿接触。
- **回合长度**：len: 182→68

### v3：修改 `crash_avoidance` → 分数 -109.6 → -115.2（-5.6）➖ 小幅退步
- **你做了什么**：唯一目标组件为 `crash_avoidance`，将其替换为 `landing_success`：当双腿接触且满足低速、直立、低角速度条件时，给予一次性正向大奖励。
- **你为什么这么做**：增加 100.0 的着陆成功奖励后，agent 将在 soft_landing 的引导下逐步探索到安全着陆状态，一旦成功即获得巨大正奖励，从而使平均分数由负转正；下一轮将观察到 landing_success 的 active_rate > 0 且 episode_sum_mean > 0，scor...
- **回合长度**：len: 68→68

### v4：修改 `landing_success` → 分数 -115.2 → -111.1（+4.1）➖ 小幅改善
- **你做了什么**：唯一目标组件为 landing_success，将其替换为 contact_reward（每步当 left_contact_next>0.5 且 right_contact_next>0.5 时给予 +1.0），并移除 crash_avoi...
- **你为什么这么做**：连续接触奖励会为 agent 提供直接梯度，促使其尽快放下双腿接触垫面，同时 soft_landing 仍会惩罚高速/倾斜，从而引导它学会在接触后继续减速并保持稳定，最终提高成功着陆率并拉升 score。
- **回合长度**：len: 68→68

### v5：修改 `contact_reward` → 分数 -111.1 → -119.0（-7.9）➖ 小幅退步
- **你做了什么**：唯一目标组件contact_reward，将其替换为landing_success，当同时满足双腿接触、低速、直立、低角速度条件时给予一次性+100奖励。
- **你为什么这么做**：一旦landing_success可被触发，代理将获得巨大成正奖励，远超soft_landing负惩罚，从而学会在接近垫面时主动减速并维持姿态，进而提高成功着陆率并拉升score。
- **回合长度**：len: 68→68

### v6：修改 `landing_success` → 分数 -119.0 → -108.2（+10.8）➖ 小幅改善
- **你做了什么**：基于v2代码，把crash_avoidance（几乎不触发）替换为landing_comfort，该组件在双腿接触时根据速度和姿态误差给出连续奖励，最大每步1.0，从而提供逐步改善的梯度。
- **你为什么这么做**：加入连续舒适奖励后，agent将在双腿接触后为获取更高奖励而主动降低速度、减小倾斜和角速度，从而提升成功着陆率，使score上升并突破-109.58。
- **回合长度**：len: 68→68

## 你的修改轨迹（按组件分组）

- **`landing_success`**：修改了 **2 次**，分数变化：v4(+4) → v6(+11)
- **`soft_landing`**：修改了 **1 次**，分数变化：v2(+105)
- **`crash_avoidance`**：修改了 **1 次**，分数变化：v3(-6)
- **`contact_reward`**：修改了 **1 次**，分数变化：v5(-8)

**在进入第 1 步诊断之前，先通读上面的修改自传。问自己：我过去改了什么？哪些成功了？我是不是在重复自己？**

# 3. 上一轮奖励函数代码（该轮得分: 255.276942）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测
    x, y = obs[0], obs[1]
    # 下一观测
    nx, ny = next_obs[0], next_obs[1]
    nvx, nvy = next_obs[2], next_obs[3]
    nangle = abs(next_obs[4])          # body_angle
    nangvel = abs(next_obs[5])         # angular_velocity
    nleft = next_obs[6]
    nright = next_obs[7]

    # 主进展信号：与目标着陆垫的距离变化
    d_current = (x**2 + y**2)**0.5
    d_next = (nx**2 + ny**2)**0.5 + 1e-8   # 避免除零
    progress = d_current - d_next          # >0 表示接近目标

    # soft health gate —— 在姿态、速度、角速度恶化时衰减正向进展
    # 使用指数衰减，温度系数根据任务特性设定
    angle_gate   = 2.718281828**(-2.0 * nangle**2)                       # 身体倾角过大时衰减
    v_sq = nvx**2 + nvy**2
    vel_gate     = 2.718281828**(-1.0 * v_sq / (1.0 + d_next))           # 高速时衰减，距离越近越严格
    angvel_gate  = 2.718281828**(-3.0 * nangvel)                         # 角速度过大时衰减

    gate = angle_gate * vel_gate * angvel_gate

    # 仅对正向进展施加 gate，负进展（远离）保留完整惩罚
    if progress > 0.0:
        progress *= gate

    # 任务完成近似信号：双腿同时接触着陆垫时给予持续性小奖励
    contact_reward = 0.2 if (nleft > 0.5 and nright > 0.5) else 0.0

    total_reward = progress + contact_reward
    components = {
        "progress": progress,
        "contact": contact_reward
    }
    return float(total_reward), components
```

# 4. Subagent 行为诊断报告

# 训练数据报告 — Iter 07

---

## 1. 行为事实

- **得分**: score=255.28, range=[164.97, 299.71]
- **终止方式**: terminated=18/20 (成功), truncated=2/20 (时间上限)
- **回合长度**: len=351.65
- **Checkpoint 轨迹**: 
  - 40%: score_mean=0.00
  - 80%: score_mean=0.00
  - (注：仅提供 40% 和 80% 两个 checkpoint 数据点)

---

## 2. 骨架演进表

| iter | score | len | term | goal_approach_ep_sum | goal_approach_active | soft_landing_ep_sum | soft_landing_active | crash_avoidance_ep_sum | crash_avoidance_active | landing_success_ep_sum | landing_success_active | contact_reward_ep_sum | contact_reward_active | landing_comfort_ep_sum | landing_comfort_active | progress_ep_sum | progress_active | contact_ep_sum | contact_active |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | -214.16 | 182.50 | 20/0/0 | 0.152 | 100.0% | 0.000 | 0.0% | 0.000 | 0.0% | — | — | — | — | — | — | — | — | — | — |
| 2 | -109.58 | 68.45 | 20/0/0 | 1.120 | 100.0% | -19.091 | 100.0% | 0.000 | 0.0% | — | — | — | — | — | — | — | — | — | — |
| 3 | -115.21 | 68.35 | 20/0/0 | 1.119 | 100.0% | -20.744 | 100.0% | — | — | 25.000 | 0.4% | — | — | — | — | — | — | — | — |
| 4 | -111.06 | 68.40 | 20/0/0 | 1.120 | 100.0% | -19.297 | 100.0% | — | — | — | — | 1.150 | 1.7% | — | — | — | — | — | — |
| 5 | -119.00 | 68.40 | 20/0/0 | 1.117 | 100.0% | -20.641 | 100.0% | — | — | 25.000 | 0.4% | — | — | — | — | — | — | — | — |
| 6 | -108.18 | 68.45 | 20/0/0 | 1.120 | 100.0% | -19.127 | 100.0% | — | — | — | — | — | — | 0.428 | 0.7% | — | — | — | — |
| **7** | **255.28** | **351.65** | **18/0/2** | — | — | — | — | — | — | — | — | — | — | — | — | 0.843 | 100.0% | 25.830 | 36.7% |

---

## 3. 数字标记

### 主导
- **contact**: magnitude_share=96.4% > 80%，为主导组件。

### 死组件
- 无。progress active_rate=100.0%，contact active_rate=36.7%，均不低于 2%。

### 淹没
- **progress**: active_rate=100.0% > 90%，但 magnitude_share=3.6% < 3%（接近但未低于 3% 阈值；标记为边界情况，不正式判定为淹没）。

### 全覆盖惩罚
- 无。contact 的 signed_share 和 magnitude_share 均为正。

### 量级对比（每步均值 = ep_sum ÷ len）
- contact: 25.830 / 351.65 ≈ **0.0735**
- progress: 0.843 / 351.65 ≈ **0.0024**

本迭代仅有两个组件，contact 每步均值最大，progress 每步均值最小。

### 修改变化（相对于 Iter 06）
Iter 07 的组件体系与 Iter 06 完全不同（从 `goal_approach + soft_landing + landing_comfort` 重构为 `progress + contact`），无法直接逐组件对比 ep_sum 和 active_rate 的 Δ。

---

## 4. 语义演化（跨轮组件功能追踪）

### 功能分组分析

#### 1. 着陆/接触信号
| 迭代 | 组件名 | 形态 | 方向 | ep_sum | active_rate |
|:---:|:---|:---|:---:|---:|---:|
| 1 | crash_avoidance | 稀疏惩罚 (一步 -10) | 负 | 0.000 | 0.0% |
| 2 | crash_avoidance | 稀疏惩罚 (一步 -10) | 负 | 0.000 | 0.0% |
| 3 | landing_success | 稀疏bonus (一步 +100, 严条件) | 正 | 25.000 | 0.4% |
| 4 | contact_reward | 连续奖励 (每步 +1, 仅双腿接触) | 正 | 1.150 | 1.7% |
| 5 | landing_success | 稀疏bonus (一步 +100, 宽条件) | 正 | 25.000 | 0.4% |
| 6 | landing_comfort | 连续奖励 (每步 max+1, 稳定性缩放) | 正 | 0.428 | 0.7% |
| **7** | **contact** | **连续奖励 (每步 +0.2, 仅双腿接触)** | **正** | **25.830** | **36.7%** |

**方向切换**：稀疏bonus ↔ 连续奖励：4 次切换（Iter 3 landing_success 稀疏 → Iter 4 contact_reward 连续 → Iter 5 landing_success 稀疏 → Iter 6 landing_comfort 连续）。Iter 7 维持连续奖励形态，但增加 gated progress 作为主导前导信号后，contact 的 active_rate 从 ~1% 跃升至 36.7%。

#### 2. 接近/进度信号
| 迭代 | 组件名 | 形态 | 方向 | ep_sum | active_rate |
|:---:|:---|:---|:---:|---:|---:|
| 1 | goal_approach | 连续 (delta 距离) | 正 | 0.152 | 100.0% |
| 2 | goal_approach | 连续 (delta 距离) | 正 | 1.120 | 100.0% |
| 3 | goal_approach | 连续 (delta 距离) | 正 | 1.119 | 100.0% |
| 4 | goal_approach | 连续 (delta 距离) | 正 | 1.120 | 100.0% |
| 5 | goal_approach | 连续 (delta 距离) | 正 | 1.117 | 100.0% |
| 6 | goal_approach | 连续 (delta 距离) | 正 | 1.120 | 100.0% |
| **7** | **progress** | **连续 (delta 距离, health-gated)** | **正** | **0.843** | **100.0%** |

**方向切换**：Iter 1-6 均为纯 delta 距离形态；Iter 7 引入指数 health gate（angle_gate × vel_gate × angvel_gate），仅在 progress>0 时衰减。形态上从"纯改善量"转向"门控改善量"：1 次切换。ep_sum 从 ~1.12 降至 0.843，但配合 contact 激活后成功实现 task solved。

#### 3. 稳定性/约束信号
| 迭代 | 组件名 | 形态 | 方向 | ep_sum | active_rate |
|:---:|:---|:---|:---:|---:|---:|
| 1 | soft_landing | 低空门控二次惩罚 | 负 | 0.000 | 0.0% |
| 2 | soft_landing | 全高度线性惩罚 | 负 | -19.091 | 100.0% |
| 3 | soft_landing | 全高度线性惩罚 | 负 | -20.744 | 100.0% |
| 4 | soft_landing | 全高度线性惩罚 | 负 | -19.297 | 100.0% |
| 5 | soft_landing | 全高度线性惩罚 | 负 | -20.641 | 100.0% |
| 6 | soft_landing | 全高度线性惩罚 | 负 | -19.127 | 100.0% |
| **7** | — | **无独立稳定性组件** | — | — | — |

**方向切换**：Iter 1 门控二次 → Iter 2 全高度线性：1 次切换。Iter 7 完全移除了独立稳定性惩罚组件，改用 health gate 内嵌于 progress 中隐式约束 stability（通过 angle_gate/vel_gate/angvel_gate 衰减正向进展）。这意味着稳定性功能从"独立显式惩罚"切换为"隐式门控"：1 次切换。

---

### 总结
Iter 07 通过将历史迭代中的三大功能（接近、着陆、稳定性）压缩为两个组件（progress + contact），实现了 score 从负值到 255.28 的跨越。稳定性功能不再独立存在，而是被吸收进 progress 的 exponential health gate 中。着陆信号从迭代 3-6 的多次稀疏/连续振荡中稳定为低值连续奖励（0.2/step），配合 gated progress 后 active_rate 显著提升。

# 4. 本轮修改的逐组件效果（编辑前 → 编辑后）

| 组件 | 编辑前均值 | 编辑后均值 | 均值Δ | 编辑前激活率 | 编辑后激活率 | 激活率Δ |
|---|---:|---:|---:|---:|---:|---:|
| contact | — (新增) | 0.111 | 新增 | — | 55.3% | — |
| goal_approach | 0.016 | — (新增) | 已删除 | 100.0% | — | — |
| landing_comfort | 0.006 | — (新增) | 已删除 | 0.7% | — | — |
| progress | — (新增) | 0.001 | 新增 | — | 100.0% | — |
| soft_landing | -0.280 | — (新增) | 已删除 | 100.0% | — | — |

| 回合级指标 | 编辑前 | 编辑后 | Δ |
|---|---:|---:|---:|
| 开发得分 | -108.18 | 255.28 | +363.45 |
| 平均回合长度 | 68.5 | 351.6 | +283.2 |
| 终止回合数 | 20/20 | 18/20 | -2 |

# 6. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=255.276942, len=351.650000, terminated=18/20, truncated=2/20, reward_errors=0
score_range=[164.967665, 299.707178]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| contact | 25.830000 | 96.4% | 96.4% | 36.7% |
| progress | 0.843410 | 3.1% | 3.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 7. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
主目标：控制一个2D飞行器从视口顶部中央区域出发，精准降落到中央目标垫上，并保持稳定、安全的接触（两条支撑腿同时着垫、机体姿势竖直、速度极低）。  
次目标：在确保安全着陆的前提下，尽可能缩短总耗时并减少引擎燃料消耗。  
不应混淆的目标：不要单纯追求最快到达而牺牲着陆安全性；也不要只追求零燃料而无限盘旋。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（或 float32，但观察含浮点标志位）
- obs[0]: x_position, 飞行器相对于目标垫的水平坐标 (m)，reward_usable: true
- obs[1]: y_position, 飞行器相对于垫面高度的垂直坐标 (m)，reward_usable: true
- obs[2]: x_velocity, 水平线速度 (m/s)，reward_usable: true
- obs[3]: y_velocity, 垂直线速度 (m/s)，reward_usable: true
- obs[4]: body_angle, 机体偏转角 (rad)，reward_usable: true
- obs[5]: angular_velocity, 角速度 (rad/s)，reward_usable: true
- obs[6]: left_support_contact, 左支撑腿接触标志 (0.0/1.0)，reward_usable: true
- obs[7]: right_support_contact, 右支撑腿接触标志 (0.0/1.0)，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: do nothing (no_engine) – 不点火，仅靠惯性运动
- action 1: fire left orientation engine – 产生顺时针力矩，改变姿态
- action 2: fire main engine – 产生向上的推力（同时可能产生微小水平分力？此处按垂直主推力理解）
- action 3: fire right orientation engine – 产生逆时针力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 两腿同时接触目标垫且机体姿态、速度满足稳定条件 → 触发 body_not_awake_or_settled，可能伴随低速度/低角速度，判为成功。
- failure-like termination: ① 机体任何部分与地面或非垫区域接触 (crash_or_body_contact)；② 水平位置超出视口边界 (horizontal_position_outside_viewport)；③ 在未安全着陆的情况下触发休眠 (body_not_awake_or_settled) 也可能失败。
- ambiguous termination: 当 body_not_awake_or_settled 的触发阈值较宽时，可能涵盖成功着陆（两腿着垫、速度低）与卡在边缘/侧翻但速度为零的情况，需由观察进一步区分。
- truncation: 未观察到显式截断，False 表示不使用截断，episode 总是由上述条件终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} (info 为空字典，不可使用)
- forbidden_or_uncertain_info_fields: 任何 info 字段均禁止使用（因为未声明）。

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1])
- velocity: x_velocity (obs[2]), y_velocity (obs[3])
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: current action (动作索引或 one-hot)，可计算是否使用主引擎/姿态引擎
- other: 无

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | crash_avoidance + goal_approach + soft_landing | -214.16 | -214.16 | 0.00 | 182.50 | crash_avoidance=-0.001 goal_approach=0.001 soft_landing=-0.009 | new_best |
| 2 | crash_avoidance + goal_approach + soft_landing | -109.58 | -109.58 | 0.00 | 68.45 | crash_avoidance=-0.002 goal_approach=0.016 soft_landing=-0.282 | new_best |
| 3 | goal_approach + landing_success + soft_landing | -115.21 | -109.58 | -5.63 | 68.35 | goal_approach=0.016 landing_success=0.536 soft_landing=-0.290 | no_meaningful_improvement |
| 4 | contact_reward + goal_approach + soft_landing | -111.06 | -109.58 | -1.49 | 68.40 | contact_reward=0.017 goal_approach=0.016 soft_landing=-0.281 | no_meaningful_improvement |
| 5 | goal_approach + landing_success + soft_landing | -119.00 | -109.58 | -9.43 | 68.40 | goal_approach=0.016 landing_success=0.643 soft_landing=-0.288 | unsolved_stagnation_fresh_restart |
| 6 | goal_approach + landing_comfort + soft_landing | -108.18 | -108.18 | 0.00 | 68.45 | goal_approach=0.016 landing_comfort=0.006 soft_landing=-0.280 | unsolved_stagnation_fresh_restart |
| 7 | contact + progress | 255.28 | 255.28 | 0.00 | 351.65 | contact=0.111 progress=0.001 | target_solved_new_best |

```
