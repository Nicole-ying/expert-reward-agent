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
- target_score: 6000.000000
- current_score: -16.602316
- gap_to_target: 6016.602316
- target_achievement_ratio: -0.277%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -16.602316）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主学习信号：奖励沿前进方向的速度
    w_forward = 1.0
    forward_velocity = next_obs[13]  # 前进速度（主体x方向速度）
    forward_reward = w_forward * forward_velocity

    # 约束1：惩罚横向运动，鼓励直线前进
    w_lateral = 0.5
    lateral_velocity = next_obs[14]  # 横向速度
    lateral_penalty = -w_lateral * (lateral_velocity ** 2)

    # 约束2：惩罚过高的角速度，保持姿态稳定
    w_angular = 0.5
    roll_rate = next_obs[16]
    pitch_rate = next_obs[17]
    yaw_rate = next_obs[18]
    angular_penalty = -w_angular * (roll_rate ** 2 + pitch_rate ** 2 + yaw_rate ** 2)

    # 约束3：惩罚剧烈的垂直运动，防止跳跃/坠落
    w_vertical = 0.1
    vertical_velocity = next_obs[15]  # 垂直速度
    vertical_penalty = -w_vertical * abs(vertical_velocity)

    total_reward = forward_reward + lateral_penalty + angular_penalty + vertical_penalty

    components = {
        "forward_velocity_reward": forward_reward,
        "lateral_velocity_penalty": lateral_penalty,
        "angular_velocity_penalty": angular_penalty,
        "vertical_velocity_penalty": vertical_penalty,
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-16.602316, len=11.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-25.541189, -7.891307]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| angular_velocity_penalty | -21.074904 | -71.4% | 71.4% | 100.0% |
| forward_velocity_reward | 0.954357 | 3.2% | 17.1% | 100.0% |
| vertical_velocity_penalty | -2.353788 | -8.0% | 8.0% | 100.0% |
| lateral_velocity_penalty | -1.037845 | -3.5% | 3.5% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个 3D 四足机器人（8 个扭矩控制关节）在平坦地面上稳定地向前行走或奔跑。机器人必须保持身体高度在 0.2~1.0 的范围内；超出该范围（摔倒或蹦飞）或出现数值异常将立即终止回合。核心目标是实现持续、稳定的前进运动，而非仅仅保持站立。

## 3. 观察空间 observation_space
- type: Box
- shape: (105,)
- dtype: float32
- obs[0]: 主体垂直高度 (body_z)
- obs[1]: 主体姿态四元数第 1 分量 (body_orientation)
- obs[2]: 主体姿态四元数第 2 分量
- obs[3]: 主体姿态四元数第 3 分量
- obs[4]: 主体姿态四元数第 4 分量
- obs[5]: 关节 1 角度 (hip_1)
- obs[6]: 关节 2 角度 (ankle_1)
- obs[7]: 关节 3 角度 (hip_2)
- obs[8]: 关节 4 角度 (ankle_2)
- obs[9]: 关节 5 角度 (hip_3)
- obs[10]: 关节 6 角度 (ankle_3)
- obs[11]: 关节 7 角度 (hip_4)
- obs[12]: 关节 8 角度 (ankle_4)
- obs[13]: 主体世界 x 方向速度（前进速度）
- obs[14]: 主体世界 y 方向速度（横向速度）
- obs[15]: 主体垂直速度 (body_z_velocity)
- obs[16]: 主体滚转角速度 (roll)
- obs[17]: 主体俯仰角速度 (pitch)
- obs[18]: 主体偏航角速度 (yaw)
- obs[19]: 关节 1 角速度
- obs[20]: 关节 2 角速度
- obs[21]: 关节 3 角速度
- obs[22]: 关节 4 角速度
- obs[23]: 关节 5 角速度
- obs[24]: 关节 6 角速度
- obs[25]: 关节 7 角速度
- obs[26]: 关节 8 角速度
- obs[27:33]: 躯干接触力/力矩 (6D)
- obs[33:39]: 左前腿上段接触力/力矩
- obs[39:45]: 左前腿下段接触力/力矩
- obs[45:51]: 左前足接触力/力矩（强地面接触信号）
- obs[51:57]: 右前腿上段接触力/力矩
- obs[57:63]: 右前腿下段接触力/力矩
- obs[63:69]: 右前足接触力/力矩（强地面接触信号）
- obs[69:75]: 左后腿上段接触力/力矩
- obs[75:81]: 左后腿下段接触力/力矩
- obs[81:87]: 左后足接触力/力矩（强地面接触信号）
- obs[87:93]: 右后腿上段接触力/力矩
- obs[93:99]: 右后腿下段接触力/力矩
- obs[99:105]: 右后足接触力/力矩（强地面接触信号）

## 4. 动作空间 action_space
- type: Box
- shape: (8,)
- dtype: float32
- bounds: [-1.0, 1.0] 每个维度
- action 0: 髋关节 1 力矩 (hip_1_torque)
- action 1: 踝关节 1 力矩 (ankle_1_torque)
- action 2: 髋关节 2 力矩 (hip_2_torque)
- action 3: 踝关节 2 力矩 (ankle_2_torque)
- action 4: 髋关节 3 力矩 (hip_3_torque)
- action 5: 踝关节 3 力矩 (ankle_3_torque)
- action 6: 髋关节 4 力矩 (hip_4_torque)
- action 7: 踝关节 4 力矩 (ankle_4_torque)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无（本环境没有明确的成功到达目标，只有通过截断结束一个 episode）
- failure-like termination:  
  a) 身体高度超出健康范围 (body_height_outside_healthy_range)，即 body_z ≤ 0.2 或 body_z ≥ 1.0  
  b) 任何状态值出现 NaN 或无穷大 (state_value_outside_finite_range)
- ambiguous termination: 无
- truncation: 到达预设的时间步上限（time_limit_reached），视为正常结束

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典，没有任何允许字段）
- forbidden_or_uncertain_info_fields: 所有与官方奖励相关的字段（如 reward_forward, reward_ctrl, reward_contact, reward_survive）以及 x_position, y_position, distance_from_origin, x_velocity, y_velocity 等均被禁止，不可使用

## 7. 可用于奖励函数的信号
- position: 主体高度 (obs[0])，主体姿态四元数 (obs[1:5])，各关节角度 (obs[5:13])
- velocity: 前进速度 (obs[13])，横向速度 (obs[14])，垂直速度 (obs[15])，滚转/俯仰/偏航角速度 (obs[16:18])，各关节角速度 (obs[19:27])
- orientation: 四元数姿态，可用于惩罚大幅度翻滚或俯仰
- contact: 各足端接触力 (obs[45:51], obs[63:69], obs[81:87], obs[99:105])，可推断足部着地状态与地面反作用力
- action/engine: 关节力矩动作本身 (action[0:8])，可用于惩罚过大的控制量

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_velocity_penalty + forward_velocity_reward + lateral_velocity_penalty + vertical_velocity_penalty | -16.60 | -16.60 | 0.00 | 11.75 | angular_velocity_penalty=-2.262 forward_velocity_reward=0.041 lateral_velocity_penalty=-0.145 vertical_velocity_penalty=-0.076 | new_best |

```
