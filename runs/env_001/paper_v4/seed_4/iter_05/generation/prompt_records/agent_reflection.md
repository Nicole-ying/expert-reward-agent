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

## 1. 行为与历史诊断

拿到训练反馈后，必须先明确回答原有三个问题：

1. **这个agent发生了什么？** episode相对正常长度明显偏短，可能在快速失败；episode很长但得分差，可能在徘徊；得分已经好但某组件尺度异常，可能存在exploit，必须用行为证据验证。
2. **哪个组件最值得干预？** 结合组件数学形态、episode_sum_mean、signed_share、magnitude_share、active_rate、外部score和episode_length判断，不得把数值占比直接写成因果贡献。
3. **我之前改了什么？** 从Agent Memory检查上一轮动作、预测和实际效果。如果上次改了A但得分没有实质变化，这次不要再次修改A。

随后确认当前失败是否来自该组件或某个缺失职责，一次只选择一个干预目标。current明显差于best时，以best代码为基础，但必须做一个新的、有证据的修改，不能原样复制best。

## 2. 信号完备性检查

检查当前奖励是否具有任务所需且可达的职责，而不是要求固定组件名称：

- 任务进展或可学习的过程引导；
- 必要的稳定、安全或动作约束；
- 当过程最优不等于任务完成时，能区分联合满足或完成状态的信号。

如果必要职责缺失、active_rate接近0、数学形态使反馈塌缩，或proxy与外部任务明显错位，进入Level 2。若职责基本完备、符号与数学形态合理，只是相对尺度异常，先进入Level 1。

## 3. 选择干预层级

### Level 1：尺度修复

适用条件：组件职责、符号和数学形态合理，证据主要表明单个组件过强或过弱。

- 只调整一个组件的系数，其他组件保持不变。
- 对同为逐步稠密信号、且progress明确承担主引导职责的早期学习阶段，旧实验中的`ratio_to_progress`思想仍然适用：`|penalty/progress| > 0.5`可作为优先检查并尝试降系数的经验触发器；`penalty/progress ≈ 0.1`可作为轻约束起点。
- 上述比例是v6实验中的有效经验先验，不是因果结论或通用硬阈值；事件bonus、持续状态奖励和正负抵消严重的差分均值不能直接套用。
- 若一次明确的尺度修复后，尺度异常已经消失但外部行为没有实质改善，不继续反复调同一系数，转Level 2。

### Level 2：有方向的数学结构变换

适用条件：必要信号缺失/不可达，或证据直接否定当前数学形态，或Level 1已修复尺度但策略仍失败。每轮只改变一个目标组件；改变该组件形态时同步设置与新值域匹配的系数，仍算一次组件干预。

| 证据模式 | 结构变换 | 下一轮验证 |
|---|---|---|
| 任务事件几乎不触发，缺少局部反馈 | sparse_to_dense：稀疏事件→连续过程证据 | active_rate及外部表现改善，且不产生proxy徘徊 |
| 极端值支配奖励 | unbounded_to_bounded：无界→归一化有界 | 极端轨迹支配下降，得分方差下降 |
| 占据好状态即可持续获奖 | state_to_improvement：状态值→状态改善量/有效势能差 | 停留不再积累收益，任务进展改善 |
| 约束在无关阶段妨碍探索 | global_to_local_gate：全局→阶段相关/局部门控 | 早期探索与局部约束同时改善；先确认不是单纯尺度过强 |
| 独立目标可互相补偿 | independent_to_joint：加权和→联合满足 | 单项刷分减少，必要条件共同改善 |
| 多个小因子相乘导致塌缩 | product_to_noncollapsing_joint：乘积→几何平均/软最小/门控和 | 非零反馈增多且联合约束保留 |
| 持续事件被重复领取 | persistent_to_transition_event：持续状态→有效状态转移 | 重复积累下降，外部完成保持或改善 |
| proxy提高但外部任务不升 | proxy_to_completion_alignment：代理目标→任务完成对齐 | proxy与外部分数重新同向 |
| 复杂耦合无法诊断 | coupled_to_diagnostic_components：耦合→少量直接组件 | 组件可解释并形成单一干预假设 |
| 稠密proxy形成中等分平台 | dense_to_task_event：全程proxy→局部/转移任务信号 | 刷新best，完成相关行为增加 |

常用数学性质：二值稀疏条件信用分配困难；连续乘积表达联合满足但可能塌缩；加权和反馈稠密但允许目标补偿；bounded函数限制极端值但输入仍需按环境尺度归一化；门控只应在证据表明“作用阶段错误”时使用。

### Level 3：重建主骨架

满足任一条件时停止局部修补：

- 同一骨架家族已迭代2轮以上，且历史最佳得分仍未超过target的25%；
- 同一结构家族连续2轮以上未刷新best，且至少做过一次Level 2；
- Level 2改变数学形态后没有实质改善；
- 同一结构家族连续3轮未刷新best且仍未达到target，即使已超过target的25%也要警惕中等分平台。

Level 3可以更换主信号框架或重新组合少量组件。expert_reward_context中的骨架是设计原语和风险提示，不是封闭候选列表；可以采用、组合、变形或基于环境事实创建新结构。

# 工具

核心Level判断必须依靠本Prompt完成。仅在根因不确定、多个Level 2变换难以区分或需要骨架细节时调用一次最相关工具：

- `search_reward_design_knowledge(query)`：检索相似失败模式和经验修复。
- `get_skeleton_detail(skeleton_name)`：查看数学形态、原理和陷阱。
- `get_reward_transformation(query)`：查看结构变换的原理、风险和验证目标。

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
# Search objective
- target_score: 200.000000
- current_score: 206.135560
- gap_to_target: -6.135560
- target_achievement_ratio: 103.068%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 206.135560）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    x, y = obs[0], obs[1]
    vx, vy = obs[2], obs[3]
    angle = obs[4]
    angvel = obs[5]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_angvel = next_obs[5]
    next_left = next_obs[6]
    next_right = next_obs[7]

    dist = (x ** 2 + y ** 2) ** 0.5
    next_dist = (next_x ** 2 + next_y ** 2) ** 0.5

    # ------------------ 1. Progress reward ------------------
    w_progress = 2.0
    progress = dist - next_dist
    progress_reward = w_progress * progress

    # ------------------ 2. Stability penalty (gated) ------------------
    gate = 1.0 / (1.0 + dist)
    w_vel = 0.2
    w_ang = 0.2
    stability_error = w_vel * (vx ** 2 + vy ** 2) + w_ang * (angle ** 2 + angvel ** 2)
    stability_penalty = -gate * stability_error

    # ------------------ 3. Fuel penalty ------------------
    w_fuel = 0.02
    fuel_penalty = -w_fuel if action != 0 else 0.0

    # ------------------ 4. Safe contact reward (sustained quality-gated) ------------------
    w_contact = 1.0
    contact_now = next_left * next_right                              # 1.0 only if both legs touch
    dist_factor = 1.0 / (1.0 + 10.0 * next_dist)                      # near center
    vel_factor = 1.0 / (1.0 + 5.0 * (next_vx ** 2 + next_vy ** 2))   # low velocity gating
    angle_factor = 1.0 / (1.0 + 5.0 * (abs(next_angle) + abs(next_angvel)))  # upright + still
    safe_contact_reward = w_contact * contact_now * dist_factor * vel_factor * angle_factor

    total_reward = progress_reward + stability_penalty + fuel_penalty + safe_contact_reward

    reward_components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,
        "fuel_penalty": fuel_penalty,
        "safe_contact_reward": safe_contact_reward
    }

    return float(total_reward), reward_components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=206.135560, len=681.450000, terminated=10/20, truncated=10/20, reward_errors=0
score_range=[127.443360, 288.929483]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| safe_contact_reward | 294.208533 | 94.6% | 94.6% | 51.4% |
| fuel_penalty | -9.771000 | -3.1% | 3.1% | 71.7% |
| stability_penalty | -4.337368 | -1.4% | 1.4% | 100.0% |
| progress_reward | 2.768706 | 0.9% | 0.9% | 99.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 载具式着陆任务。主体从视口上方中央附近开始，带有随机初速扰动。**核心目标**是在尽可能短的时间内安全抵达画面中央的目标着陆垫并稳定停驻，同时尽可能减少引擎推力消耗。智能体必须学会精确控制位置与速度，在接近着陆垫时减速、保持竖直姿态，并实现两条支撑腿的平稳接触。

**次要目标**：节约引擎燃料；快速完成任务。  
**不应混淆的目标**：不存在与到达目标同等权重的冲突目标（如“保持高速”或“探索未知区域”），燃料节省仅为附属要求。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（来自 Box 观察）
- 各维度含义：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标垫的水平坐标 | true |
| 1 | y_position | 相对于目标垫高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 主体姿态角（绝对值或从竖直偏移） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左支撑腿是否触地（1.0 或 0.0） | true（谨慎） |
| 7 | right_support_contact | 右支撑腿是否触地（1.0 或 0.0） | true（谨慎） |

**注意**：左/右支撑接触标志可用于奖励，但需考虑它们可能与导致终止的“crash_or_body_contact”混淆风险（见下文分析）。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 各动作含义：

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不点火，仅靠惯性漂移 |
| 1 | left_orientation_engine | 点燃左侧姿态调节引擎，产生一个方向扭矩/推力 |
| 2 | main_engine | 点燃主引擎（推测向上推力，对抗重力） |
| 3 | right_orientation_engine | 点燃右侧姿态调节引擎，产生相反方向扭矩/推力 |

动作空间为离散选择，每次步只能执行四个动作之一。无连续控制。

## 5. step 与终止条件分析
### 5.1 终止模式
源码中的终止判断为：
```python
terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
```
其中 **crash_or_body_contact**、**horizontal_position_outside_viewport** 和 **body_not_awake_or_settled** 均为复合条件（具体实现未暴露）。

- **success-like termination**：`body_not_awake_or_settled` 可能表示主体已安定在着陆垫上（无运动或规定时间内稳定），此类终止可视为成功。
- **failure-like termination**：`crash_or_body_contact`（如翻滚、身体直接碰撞）与 `horizontal_position_outside_viewport`（飞出边界）明显是失败。
- **ambiguous termination**：`body_not_awake_or_settled` 若发生在着陆垫之外，可能不算成功，但由于无额外信息，无法在 step 返回值中分辨。环境没有提供显式的“成功 / 失败”标志。
- **truncation**：源码中 `step` 未返回 `truncated`（第四个返回值为 `False`），因此无时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: `info` 固定为空字典 `{}`，无任何可用字段。
- forbidden_or_uncertain_info_fields: 不可假设存在 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等。`info` 完全不可用。

## 7. 可用于奖励函数的信号
基于观察和动作的可用信号分类如下：

- **位置**：`x_position`（0）、`y_position`（1）—— 相对目标垫坐标，目标为 (0, 0)。
- **速度**：`x_velocity`（2）、`y_velocity`（3）—— 目标为零。
- **方向/姿态**：`body_angle`（4）、`angular_velocity`（5）—— 理想为零（竖直稳定）。
- **接触/着陆**：`left_support_contact`（6）、`right_support_contact`（7）—— 标志双腿是否触地，可用于安全着陆奖励，但需注意避免与 crash 混淆（见下节）。
- **动作/引擎**：采取的动作 `action`（0~3）用于推力惩罚（如约束不点火为主，或限制除主引擎外的浪费）。可设计动作惩罚。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | fuel_penalty + progress_reward + safe_contact_bonus + stability_penalty | 139.53 | 139.53 | 0.00 | 1000.00 | fuel_penalty=-0.013 progress_reward=0.006 safe_contact_bonus=0.225 stability_penalty=-0.021 | new_best |
| 2 | fuel_penalty + progress_reward + safe_contact_bonus + stability_penalty | 114.50 | 139.53 | -25.03 | 907.15 | fuel_penalty=-0.016 progress_reward=0.006 safe_contact_bonus=0.490 stability_penalty=-0.021 | no_meaningful_improvement |
| 3 | fuel_penalty + progress_reward + safe_contact_reward + stability_penalty | 206.14 | 206.14 | 0.00 | 681.45 | fuel_penalty=-0.012 progress_reward=0.005 safe_contact_reward=0.403 stability_penalty=-0.019 | target_solved_new_best |
| 4 | fuel_penalty + progress_reward + safe_contact_reward + stability_penalty | 206.14 | 206.14 | 0.00 | 681.45 | fuel_penalty=-0.012 progress_reward=0.005 safe_contact_reward=0.403 stability_penalty=-0.019 | target_solved_no_improvement |

```
