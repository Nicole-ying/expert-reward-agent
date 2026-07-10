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
# Duplicate reward retry
The previous generation duplicated iter 1 (runs/env_002/paper_bipedal_v4/seed_0/iter_01/generation/reward_v1.py). Retry 2: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python

```

# Search objective
- target_score: 300.000000
- current_score: 300.224778
- gap_to_target: -0.224778
- target_achievement_ratio: 100.075%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 300.224778）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 前进速度奖励（只奖励正向移动）
    hor_vel = next_obs[2]
    forward_reward = 2.0 * max(0.0, hor_vel)

    # 2. 平衡/稳定惩罚：身体倾角与角速度的平方惩罚
    angle = next_obs[0]
    ang_vel = next_obs[1]
    stability_penalty = -2.0 * (angle ** 2) - 0.5 * (ang_vel ** 2)

    # 3. 节能惩罚：关节力矩平方和
    energy_penalty = -0.005 * (action[0] ** 2 + action[1] ** 2 +
                               action[2] ** 2 + action[3] ** 2)

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_velocity": forward_reward,
        "stability": stability_penalty,
        "energy": energy_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=300.224778, len=1078.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[299.070953, 301.485751]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity | 1008.781804 | 95.1% | 95.1% | 99.3% |
| stability | -43.264310 | -4.1% | 4.1% | 100.0% |
| energy | -8.251407 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境要求一个平面双足身体在起伏不平的地形上持续前进。主目标是**尽可能远、尽可能快地向前移动**，同时**尽可能减少能量消耗**（动作幅度/力矩）。次要目标是**维持上身稳定、避免摔倒**，因为一旦身体倾覆或到达地形尽头，回合即终止。因此奖励设计需引导智能体学习高效、平稳的前进步态，而不是到达某个固定终点。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断）
- 各维度说明与 reward_usable 标记：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | hull_angle | 主体相对于竖直方向的倾角（越大越倾斜） | true |
| 1 | hull_angular_velocity | 主体倾角的角速度 | true |
| 2 | horizontal_velocity | 前进/后退方向的线速度（正值=向前） | true |
| 3 | vertical_velocity | 上下方向线速度（正值=上升） | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1是否触地（1.0=触地，0.0=悬空） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
| 10 | hip2_speed | 腿2髋关节角速度 | true |
| 11 | knee2_angle | 腿2膝关节角度 | true |
| 12 | knee2_speed | 腿2膝关节角速度 | true |
| 13 | leg2_contact | 腿2是否触地 | true |
| 14–23 | lidar_0 … lidar_9 | 10个激光雷达测距值，描述前方地形轮廓 | true |

所有维度均可作为奖励信号来源，但雷达数据仅反映地形几何，直接用于奖励需要谨慎映射（如惩罚前方陡坡）。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续，各维范围 [-1.0, 1.0]
- 动作维度含义：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | hip_torque_leg1 | 施加在腿1髋关节上的力矩 |
| 1 | knee_torque_leg1 | 施加在腿1膝关节上的力矩 |
| 2 | hip_torque_leg2 | 施加在腿2髋关节上的力矩 |
| 3 | knee_torque_leg2 | 施加在腿2膝关节上的力矩 |

动作幅度的平方和可作为能量消耗的代理指标。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` – 走完前方全部地形，通常代表任务圆满达成。
- **failure-like termination**：`body_fallen_over` – 身体倾覆，失去平衡，代表失败。
- **ambiguous termination**：默认 `terminated` 为布尔值，在 step 返回时无法区分上述两种原因。
- **truncation**：未提及超时截断，推测由环境内部最大步数控制，但在本次 `step` 接口中未见传递。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无法获取明确的成功标记）
- **explicit_failure_flag_available**: false  
  （同上）
- **allowed_info_fields**：无（info = {}）
- **forbidden_or_uncertain_info_fields**：全部 info 子字段均为不可用，不得依赖任何未声明的键。

**关键结论**：奖励函数**不可基于终止原因**来区分成功/失败（如不可给到达终点额外奖励），因为无法从 `terminated` 布尔值可靠分离两种终止情况。奖励逻辑必须全部基于状态‑动作历程的内部信号构建。

## 7. 可用于奖励函数的信号
- **位置相关**：雷达测距（lidar_*）提供前方地形相对距离，可间接反映位置与地形起伏，但不提供绝对位置（无x坐标）。
- **速度相关**：`horizontal_velocity`（前进速度，可直接奖励）、`vertical_velocity`（可用于检测跳跃、坠落或异常运动）。
- **姿态/平衡相关**：`hull_angle`、`hull_angular_velocity`，身体倾角与角速度。
- **接触相关**：`leg1_contact`、`leg2_contact`，可用于鼓励交替触地，或惩罚双腿同时离地。
- **动作/力矩相关**：`action` 各维（力矩），平方和或绝对值可作为能耗惩罚。
- **关节状态**：四条关节角度与角速度，可用于惩罚过大的关节运动或极限位置。
- **地形预览**：lidar_0 … lidar_9，可辅助判断是否为陡坡或平台，谨慎作为奖励调整项。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy + forward_velocity + stability | 300.22 | 300.22 | 0.00 | 1078.05 | energy=-0.010 forward_velocity=0.556 stability=-0.036 | target_solved_new_best |

```
