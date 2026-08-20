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
- target_score: 2000.000000
- current_score: -172.445707
- gap_to_target: 2172.445707
- target_achievement_ratio: -8.622%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -172.445707）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # --- 提取 next_obs 的状态 ---
    body_z = next_obs[0]
    quat_w = next_obs[1]
    quat_x = next_obs[2]
    quat_y = next_obs[3]
    quat_z = next_obs[4]
    # 速度
    body_x_vel = next_obs[13]
    body_y_vel = next_obs[14]
    roll_vel  = next_obs[16]
    pitch_vel = next_obs[17]
    yaw_vel   = next_obs[18]

    # --- 健康门控因子（基于 next_obs）---
    # 1. 高度健康：body_z >= 0.5 完全健康，0.5->0.2 线性衰减到 0
    if body_z >= 0.5:
        health_gate = 1.0
    elif body_z <= 0.2:
        health_gate = 0.0
    else:
        health_gate = (body_z - 0.2) / 0.3

    # 2. 直立健康
    body_up_z = 1.0 - 2.0 * (quat_x**2 + quat_y**2)
    if body_up_z >= 0.5:
        upright_gate = 1.0
    else:
        upright_gate = max(0.0, body_up_z / 0.5)

    combined_gate = health_gate * upright_gate

    # --- 奖励组件 ---
    w_fwd = 1000.0
    forward_reward = w_fwd * body_x_vel * combined_gate

    w_lat = 2.0
    lat_penalty = w_lat * body_y_vel**2

    w_ang = 0.5
    ang_penalty = w_ang * (roll_vel**2 + pitch_vel**2 + yaw_vel**2)

    total_reward = forward_reward - lat_penalty - ang_penalty

    components = {
        'forward_reward': forward_reward,
        'lat_penalty': -lat_penalty,
        'ang_penalty': -ang_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-172.445707, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-381.176492, 31.139299]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 7149.104396 | 69.0% | 99.7% | 43.9% |
| ang_penalty | -28.043761 | -0.3% | 0.3% | 4.6% |
| lat_penalty | -7.450969 | -0.1% | 0.1% | 4.5% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标  
本环境要求控制一个 3D 四足机器人，使用 8 个连续扭矩关节产生稳定的前向步态（行走或奔跑）。机器人的身体高度必须维持在健康范围（0.2 ~ 1.0）内，超出范围立即触发 episode 终止（摔倒或弹跳过高）。核心任务是在保持直立和高度约束的前提下，持续获得尽可能大的前向线速度。次要目标包括抑制横向漂移、减小能量消耗与动作振荡。本任务不应被混淆为纯平衡存活任务——即使机器人站着不动，只要高度合规就不终止，但仅站立无法获得有效的前进奖励。

## 3. 观察空间 observation_space  
- type: Box  
- shape: (27,)  
- dtype: float32（默认）  
- 详细字段（索引 0–26）：  

### 3.1 位置部分（13 维）  
| 索引 | 名称 | 含义 | 可用于奖励 | 备注 |
|------|------|------|-------------|------|
| 0 | `body_z` | 身体（基座）在世界坐标系中的高度 | 是 | 直接用于高度健康约束 |
| 1 | `quat_w` | 姿态四元数实部 w | 是 | 用于计算直立程度 |
| 2 | `quat_x` | 姿态四元数 x 分量 | 是 | 同上 |
| 3 | `quat_y` | 姿态四元数 y 分量 | 是 | 同上 |
| 4 | `quat_z` | 姿态四元数 z 分量 | 是 | 同上 |
| 5 | `joint_1_angle` | 第 1 髋关节角度 | 可选 | 姿态/步态相关 |
| 6 | `joint_2_angle` | 第 1 踝关节角度 | 可选 | 同上 |
| 7 | `joint_3_angle` | 第 2 髋关节角度 | 可选 | 同上 |
| 8 | `joint_4_angle` | 第 2 踝关节角度 | 可选 | 同上 |
| 9 | `joint_5_angle` | 第 3 髋关节角度 | 可选 | 同上 |
| 10 | `joint_6_angle` | 第 3 踝关节角度 | 可选 | 同上 |
| 11 | `joint_7_angle` | 第 4 髋关节角度 | 可选 | 同上 |
| 12 | `joint_8_angle` | 第 4 踝关节角度 | 可选 | 同上 |

### 3.2 速度部分（14 维）  
| 索引 | 名称 | 含义 | 可用于奖励 | 备注 |
|------|------|------|-------------|------|
| 13 | `body_x_velocity` | 世界坐标系 x 方向线速度（前进方向） | **是（核心）** | 主要任务信号 |
| 14 | `body_y_velocity` | 世界坐标系 y 方向线速度（横向） | 是 | 抑制横向漂移 |
| 15 | `body_z_velocity` | 身体垂直速度 | 是 | 防止剧烈跳跃 |
| 16 | `body_roll_velocity` | 滚转角速度 | 是 | 惩罚翻滚 |
| 17 | `body_pitch_velocity` | 俯仰角速度 | 是 | 惩罚栽倒 |
| 18 | `body_yaw_velocity` | 偏航角速度 | 可选 | 若需保持航向 |
| 19 | `joint_1_velocity` | 第 1 髋关节角速度 | 可选 | 平滑性/能耗 |
| 20 | `joint_2_velocity` | 第 1 踝关节角速度 | 可选 | 同上 |
| 21 | `joint_3_velocity` | 第 2 髋关节角速度 | 可选 | 同上 |
| 22 | `joint_4_velocity` | 第 2 踝关节角速度 | 可选 | 同上 |
| 23 | `joint_5_velocity` | 第 3 髋关节角速度 | 可选 | 同上 |
| 24 | `joint_6_velocity` | 第 3 踝关节角速度 | 可选 | 同上 |
| 25 | `joint_7_velocity` | 第 4 髋关节角速度 | 可选 | 同上 |
| 26 | `joint_8_velocity` | 第 4 踝关节角速度 | 可选 | 同上 |

- 所有字段均可直接用于奖励，但具体是否采纳取决于职责设计。

## 4. 动作空间 action_space  
- type: Box  
- shape: (8,)  
- bounds: [-1.0, 1.0] 每关节  
- continuous: true  
- 每个动作维度的含义（与关节对应）：  

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | `hip_1_torque` | 第一个髋关节扭矩指令 |
| 1 | `ankle_1_torque` | 第一个踝关节扭矩指令 |
| 2 | `hip_2_torque` | 第二个髋关节扭矩指令 |
| 3 | `ankle_2_torque` | 第二个踝关节扭矩指令 |
| 4 | `hip_3_torque` | 第三个髋关节扭矩指令 |
| 5 | `ankle_3_torque` | 第三个踝关节扭矩指令 |
| 6 | `hip_4_torque` | 第四个髋关节扭矩指令 |
| 7 | `ankle_4_torque` | 第四个踝关节扭矩指令 |

- 动作是直接施加在关节上的力矩，可用于构建动作平滑或能耗惩罚。

## 5. step 与终止条件分析  
### 5.1 终止模式  
- **success-like termination**：无显式成功判断。环境本身不给出“到达目标/完成任务”的终止，只有时间截断（truncation）。从任务语义看，若能保持高度、存活到 episode 结束并获得持续前进速度，可视为成功的表现形式。  
- **failure-like termination**：
  - `body_height_outside_healthy_range`：身体高度 ≤ 0.2（摔倒）或 ≥ 1.0（猛烈弹跳），属于明确的失败。
  - `state_value_outside_finite_range`：任意状态变为 NaN 或无穷大，属异常/失败。
- **ambiguous termination**：无。
- **truncation**：时间限制到达，episode 自然结束。对于该任务，本身不算失败；多数成功训练下，时间截断即代表存活且前进了一段距离。

### 5.2 success/failure 信号可用性  
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **true**（通过 terminated 标志可推断为失败，但需自行判断具体原因）  
- allowed_info_fields: **[]**（为空列表）  
- forbidden_or_uncertain_info_fields:  
  - `reward_forward`、`reward_ctrl`、`reward_contact`、`reward_survive`、`x_position`、`y_position`、`distance_from_origin` 等字段明确禁止使用。  
  - 不允许使用任何未在 `allowed_info_fields` 中声明的 info 字段。

## 7. 可用于奖励函数的信号  
- **前进速度**：`next_obs[13]`（body_x_velocity），直接体现任务核心目标。  
- **高度保持**：`next_obs[0]`（body_z）处于 (0.2, 1.0) 内。  
- **直立姿态**：通过 `next_obs[1:5]`（四元数）计算身体 z 轴与世界 z 轴投影 `body_up_z = 1 - 2*(quat_x² + quat_y²)`，接近 1 表示完全竖直。  
- **横向/垂直速度抑制**：`next_obs[14]`（body_y_velocity）、`next_obs[15]`（body_z_velocity）。  
- **角速度惩罚**：`next_obs[16:18]`（roll, pitch 角速度）。  
- **动作平滑/能耗**：`action` 本身（扭矩绝对值或平方）、`obs[19:27]` 关节速度（可用于关节运动惩罚）。  
- **步态发育辅助信号**：可选用关节位置 `next_obs[5:13]` 或速度 `next_obs[19:27]`，避免僵直或过度振荡。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | -27.22 | -27.22 | 0.00 | 409.80 | angular_penalty=0.258 height_penalty=0.013 lateral_penalty=0.055 upright_penalty=0.265 velocity_reward=0.205 | new_best |
| 2 | angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | 13.82 | 13.82 | 0.00 | 29.25 | angular_penalty=0.316 height_penalty=0.623 lateral_penalty=0.065 upright_penalty=0.245 velocity_reward=0.171 | new_best |
| 3 | angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | 24.08 | 24.08 | 0.00 | 128.45 | angular_penalty=0.302 height_penalty=0.635 lateral_penalty=0.061 upright_penalty=0.254 velocity_reward=0.160 | new_best |
| 4 | angular_penalty + height_penalty + lateral_penalty + upright_penalty + velocity_reward | 26.07 | 26.07 | 0.00 | 181.85 | angular_penalty=0.266 height_penalty=0.133 lateral_penalty=0.055 upright_penalty=0.277 velocity_reward=0.185 | same_skeleton_oscillation_fresh_restart |
| 5 | ang_penalty + forward_reward + lat_penalty | -48.96 | 26.07 | -75.02 | 1000.00 | ang_penalty=-3.629 forward_reward=109.829 lat_penalty=-0.813 | no_meaningful_improvement |
| 6 | ang_penalty + forward_reward + lat_penalty | -172.45 | 26.07 | -198.51 | 1000.00 | ang_penalty=-3.624 forward_reward=114.174 lat_penalty=-0.788 | no_meaningful_improvement |

```
