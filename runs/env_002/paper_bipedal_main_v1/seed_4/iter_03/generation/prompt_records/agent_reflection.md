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
- target_score: 300.000000
- current_score: 264.385120
- gap_to_target: 35.614880
- target_achievement_ratio: 88.128%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 264.385120）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励（正值表示向前，直接作为主学习信号）
    forward_speed = next_obs[2]
    forward_reward = forward_speed  # 系数 1.0

    # 稳定性惩罚：惩罚大倾角、高角速度和明显垂直速度（弹跳）
    # 系数从 0.5/0.1/0.5 提升到 2.5/0.5/2.5（~5x），使稳定性约束更有效
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -2.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.5 * (angular_vel ** 2)
    vertical_vel_penalty = -2.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=264.385120, len=1152.700000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[262.922693, 265.753491]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.228813 | 97.9% | 97.9% | 100.0% |
| stability_penalty | -10.940911 | -2.1% | 2.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个两足机器人（两条腿，每条腿有髋关节和膝关节）在不平坦的 2D 地形上稳定行走。  
主要目标是尽可能远、尽可能快地向前移动；次要目标是尽量降低能量消耗（动作力度）。  
机器人需要协调双腿的关节力矩，生成稳定的步态。一旦身体倒下，回合立即结束。

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: float32（假设，实际未见明确定义，通常连续观测为 float）  
- 各维度含义：  
  - obs[0]: hull_angle —— 身体（主躯干）相对于竖直方向的倾斜角  
  - obs[1]: hull_angular_velocity —— 身体角速度  
  - obs[2]: horizontal_velocity —— 前向/后向水平线速度  
  - obs[3]: vertical_velocity —— 上下线速度  
  - obs[4]: hip1_angle —— 腿 1 髋关节角度  
  - obs[5]: hip1_speed —— 腿 1 髋关节角速度  
  - obs[6]: knee1_angle —— 腿 1 膝关节角度  
  - obs[7]: knee1_speed —— 腿 1 膝关节角速度  
  - obs[8]: leg1_contact —— 腿 1 接地标志（1.0=接触地面，0.0=未接触）  
  - obs[9]: hip2_angle —— 腿 2 髋关节角度  
  - obs[10]: hip2_speed —— 腿 2 髋关节角速度  
  - obs[11]: knee2_angle —— 腿 2 膝关节角度  
  - obs[12]: knee2_speed —— 腿 2 膝关节角速度  
  - obs[13]: leg2_contact —— 腿 2 接地标志（1.0=接触地面，0.0=未接触）  
  - obs[14] ~ obs[23]: lidar_0 ~ lidar_9 —— 10 个前方地形激光雷达测距值（前方不同角度的距离）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续动作，每个分量范围 [-1.0, 1.0]
- action 0: hip_torque_leg1 —— 施加于腿 1 髋关节的力矩（扭矩）
- action 1: knee_torque_leg1 —— 施加于腿 1 膝关节的力矩
- action 2: hip_torque_leg2 —— 施加于腿 2 髋关节的力矩
- action 3: knee_torque_leg2 —— 施加于腿 2 膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain` —— 机器人到达地形的终点（任务目标完成）
- failure-like termination: `body_fallen_over` —— 身体倒下（任务失败）
- ambiguous termination: 无（从 step 源码看，terminated 只由这两个条件构成，不存在其他模糊情况）
- truncation: step 返回的第四个元素为 `False`，表示不存在时间截断，所有终止都是由环境状态触发的真实终止

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  （`info` 字典为空，没有显式成功标志；只能从终止时是否为 `reached_end_of_terrain` 推断）
- explicit_failure_flag_available: false  
  （同上，需要根据是否为 `body_fallen_over` 推断）
- allowed_info_fields: 无（`info` 为空，下游 reward 函数不允许依赖任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，也不可假设存在官方 success/failure 标记

## 7. 可用于奖励函数的信号
从当前观测 `obs`、动作 `action` 和下一步观测 `next_obs` 中可以直接提取以下信号（需自行计算）：

- **前进速度**  
  - `obs[2]` / `next_obs[2]`：水平速度（正值前进，负值后退），可直接鼓励向前移动。
- **身体姿态稳定性**  
  - `obs[0]` / `next_obs[0]`：身体倾角，理想应接近 0（直立）；可惩罚大倾角。
  - `obs[1]`：身体角速度，鼓励小的角速度变化。
- **能量消耗（动作幅度）**  
  - `action` 的平方和或绝对值之和：反映当前控制力矩的大小，可用来惩罚过大动作以降低能耗。
- **关节状态与步态**  
  - `obs[4] ~ obs[7]`（腿 1 髋、膝角度/速度）、`obs[9] ~ obs[12]`（腿 2 对应值）：可设计周期性、摆动/支撑相位奖励。
  - `obs[8]` / `obs[13]`：接地标志，可用于检测步态相位（支撑相、摆动相），鼓励交替接地。
- **地形感知（激光雷达）**  
  - `obs[14:24]`：前方地形距离信息，可用于惩罚与障碍物过近的行为，但通常不作为主要前进奖励的直接来源。
- **终止时的结果推断**  
  - 虽然 `info` 为空，但可在训练循环中结合 `terminated` 标志与 `next_obs` 的状态判断：若水平速度接近零且身体倾角超过某阈值，可视为倒下；若 `terminated` 且身体直立、水平位置到达终点附近（需从环境外部获得位置信息，但此处观测中无位置，故无法直接获得终点信息），因此纯从观测难以区分成功终止与失败终止。设计奖励时需注意不要依赖终点识别。

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + stability_penalty | 103.03 | 103.03 | 0.00 | 787.75 | forward_reward=0.338 stability_penalty=-0.022 | new_best |
| 2 | forward_reward + stability_penalty | 264.39 | 264.39 | 0.00 | 1152.70 | forward_reward=0.331 stability_penalty=-0.027 | new_best |

```
