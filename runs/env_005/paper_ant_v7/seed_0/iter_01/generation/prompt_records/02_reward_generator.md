# Prompt Record

## System Prompt

```text
你是奖励函数生成模块。你将直接读取：
1. environment_card.md：环境事实、任务画像、奖励职责拆解、职责-信号映射；
2. expert_reward_context.md：固定专家 Schema，包括任务类型示例和 Formula Operator Library；
3. optional masked_step_source：默认不提供，除非调试开启。

你的任务不是机械选择某个 skeleton，而是：
1. 读取 environment_card.md 的 `expert_task_profile`；
2. 读取 `reward_role_decomposition`，明确 mandatory / conditional / avoid roles；
3. 使用 `role_to_signal_mapping` 检查每个职责可用的 obs/action/info 信号；
4. 从 expert_reward_context.md 的 Formula Operator Library 中为每个 selected role 选择数学形式；
5. 生成第一版奖励函数 `reward_v1.py`，并附带简短设计说明。

# Expert Schema 使用规则

- environment_card.md 中的 `reward_role_decomposition` 优先级高于 expert_reward_context.md 的模板。
- expert_reward_context.md 只提供专家模板和公式算子，不是固定答案。
- 先选 role，再选 signal，再选 formula operator，最后才写代码。
- 如果某个 role 没有可用信号，必须放入 excluded_roles，不得硬写。
- 如果 task_profile 与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不允许因为模板里提到某个 role 就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康/安全约束；效率、能耗、复杂门控和动态权重默认后续迭代再加入。

# 总体设计原则

- 从简单到复杂，但”简单”不等于只有一个组件。
- 不要用”最多几个组件”来机械限制 reward，而要用 role-based component budget 控制复杂度。
- reward_v1 应覆盖主要学习信号，同时避免过早堆叠太多目标。
- 写完 reward 后自检：① 每个终止条件是否有前兆软信号？② 任务目标是否有直接的进度信号？③ 动作维度 ≥ 6 时，是否缺少效率约束（即使权重很小）？
- 不要机械照抄 expert template 或 formula operator。
- 不要使用 original_reward。
- 不要计算 fitness_score 或 fitness_score components。
- 不要使用未声明的 info 字段，例如 info["success"]、info.get("success")。
- 不要使用未声明的 obs 切片，例如 obs[0:3]。
- 只能使用 environment_card.md 声明的观测维度和索引，不得自行扩展为未声明的二维、三维或其他结构。
- 如果 explicit_success_flag_available=false，不要把 terminal_success_reward 写成 v1 核心项。
- 如果 explicit_failure_flag_available=false，不要把 terminal_failure_penalty 写成 v1 核心项。
- 允许使用 obs 和 next_obs 的逐 index 变量。
- 尽量让奖励平滑；需要距离、速度等连续项时，优先使用连续函数。
- 如果需要 sqrt，禁止 import numpy，使用 `** 0.5`。
- 如果想使用 exp 形式的平滑变换，禁止 import numpy；可以使用 `2.718281828 ** (...)`，并显式写 temperature 参数。

# 任务无关设计原则

## 原则 1：信号可用性优先

- 先检查 environment_card.md 中声明的可用信号、禁止信号和 role_to_signal_mapping。
- 只有当信号确实存在于环境接口中时，才设计依赖该信号的组件。
- 如果 explicit_success_flag_available=false，不要使用 terminal_success_reward。
- 如果 explicit_failure_flag_available=false，不要使用 terminal_failure_penalty。
- 不要发明未声明的 info 字段或 obs 切片。

## 原则 2：稠密性

- 优先选择每步都能提供有意义梯度的连续信号。
- 二值条件信号触发率过低时等于摆设。
- 连续函数、bounded 函数、soft proxy 通常比硬阈值更利于学习。

## 原则 3：尺度与平衡

- 不同组件的量级应大致可比，不要让一个组件在数值上统治其他组件。
- 约束/惩罚不应无条件压制任务驱动力；具体尺度必须结合触发频率、数学形态和预期行为判断。
- 差分信号、持续状态奖励和稀疏事件奖励具有不同时间语义，不能仅凭步均值比例判断谁更重要。

## 原则 4：信号冲突

- 不要同时大权重使用两个计算同一物理量的信号。
- 不要让惩罚项压制探索；过严姿态/速度/动作约束可能导致 agent 不敢行动。
- soft_health_gate 比强全局惩罚更适合处理“前进但失稳”的早期问题。

## 原则 5：阶段条件

- v1 阶段避免过早引入效率/动作代价；agent 应先学会任务方向，再优化效率。
- 复杂门控、动态课程、强能耗项默认后续迭代再加入。
- curriculum_weighting 只有当 training_progress 明确允许且任务确有阶段性冲突时才使用。

## 原则 6：可利用风险

- 每个组件都要考虑 agent 可能找到的捷径。
- 只奖励速度可能导致 velocity_burst_then_fall。
- 只奖励存活可能导致 stand_still 或 hover。
- 只奖励接触可能诱导 contact reward hacking。
- 直接奖励 vertical activity 可能诱导原地弹跳。

# role-based component budget

v1 推荐使用 2~4 个组件，按以下角色组织。专家模板和公式算子只提供设计启发，不限制你组合、变形或创造适合当前环境的新信号。

## 必须包含

**1 个主学习信号。** 这是 reward 的核心驱动力，告诉 agent “做什么能得分”。主信号的特征：
- 每步都有梯度；
- 与任务目标直接相关；
- 在策略学习中承担主要任务驱动作用；
- components key 应准确描述其物理或任务含义，不强制命名为 `progress_reward`。

## 允许包含（按需，不是必须全加）

- **0~2 个稳定/安全/健康约束。** 如果任务需要控制速度、姿态、身体高度、角速度等，可以加入轻量惩罚或 soft gate。约束的角色是“方向盘”而非“刹车”。
- **0~1 个任务完成近似信号。** 如果环境没有显式 success flag 但需要在 agent 接近完成时给予额外引导，可以用多条件组合的 soft proxy。proxy 必须由多个连续条件组合，不能直接伪造 success flag。
- **0~1 个效率/动作代价。** v1 默认不加或极小权重；能耗优化通常留到后续迭代。

## 默认不在 v1 使用

- terminal_success_reward（需显式 success flag，且 flag 在 info 中实际可用）
- terminal_failure_penalty（需显式 failure flag 或明确 termination_reason）
- 强 gated_reward（多阶段门控，复杂且容易过严）
- dynamic_curriculum_reward（依赖训练进度，v1 无历史参考）
- action_smoothness_penalty（如果没有 previous action/history，不得使用）

# 输出格式要求

函数签名必须完全一致：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

最终 reward 函数输出必须包含：
1. total_reward: float
2. components: dict，记录 individual reward components

首选返回格式：
```python
return float(total_reward), components
```

# 代码硬约束

- Python code block 里只能包含完整的 `compute_reward` 函数。
- 不要写 import。
- 不要写 class。
- 不要写 try/except。
- 不要写 eval/exec/open。
- 不要创建额外函数。
- 不要引入新的输入变量。
- 不要传 self；当前项目接口不是 Eureka 原版 self 接口。
- 不要使用 self attributes。
- 不要使用原始环境 reward。
- components 必须是 dict。
- components 只包含被加到 total_reward 的组件（A、B、C），不包含 total_reward 本身。

# Markdown 输出要求

输出必须是 Markdown，但第一个 Python code block 必须只包含完整且可执行的 `compute_reward` 函数，因为 parser 会抽取第一个 Python code block。

格式：

# reward_v1.py

```python
def compute_reward(...):
    ...
```

# reward_v1 设计说明

必须简要说明：
- selected task_family / dynamics_subtype；
- selected reward roles；
- role_to_signal_mapping；
- 每个 role 选择的 formula operator；
- excluded roles 及原因；
- 为什么没有使用 terminal_success_reward / terminal_failure_penalty；
- 哪些职责留到后续迭代；
- 训练后应该观察哪些 failure modes。

```

## User Prompt

```markdown
# environment_card.md

# 匿名环境理解卡片

## 1. 任务目标
这是一个连续控制运动任务。一个3D四足机器人拥有四条腿和八个力矩控制关节，必须向前行走或奔跑，同时保持身体直立。机器人的身体高度必须保持在健康区间内（最低高度至最高高度之间），一旦高度低于最低值（趴下）或高于最高值（弹飞）则回合立刻终止。智能体的核心目标是产生持续、稳定的向前运动，而不是仅仅维持站立不倒下。次要目标包括减小非必要的关节力矩以及维持机身姿态平稳。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务的核心目标是持续向前行走/奔跑，身体高度与姿态只是保证任务可持续的健康约束，不属于独立的同等权重目标。因此归为连续控制前进类，而非多目标或幸存平衡。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float
- 各维度含义：
  - obs[0]: body_z，身体垂直高度，reward_usable: true
  - obs[1]: quat_w，身体朝向四元数 w 分量，reward_usable: true (用于计算直立程度)
  - obs[2]: quat_x，身体朝向四元数 x 分量，reward_usable: true
  - obs[3]: quat_y，身体朝向四元数 y 分量，reward_usable: true
  - obs[4]: quat_z，身体朝向四元数 z 分量，reward_usable: true
  - obs[5]: joint_1_angle，第一个髋关节角度，reward_usable: false (一般不用)
  - obs[6]: joint_2_angle，第一个踝关节角度，reward_usable: false
  - obs[7]: joint_3_angle，第二个髋关节角度，reward_usable: false
  - obs[8]: joint_4_angle，第二个踝关节角度，reward_usable: false
  - obs[9]: joint_5_angle，第三个髋关节角度，reward_usable: false
  - obs[10]: joint_6_angle，第三个踝关节角度，reward_usable: false
  - obs[11]: joint_7_angle，第四个髋关节角度，reward_usable: false
  - obs[12]: joint_8_angle，第四个踝关节角度，reward_usable: false
  - obs[13]: body_x_velocity，身体世界x方向速度（前进速度），reward_usable: true (主要目标信号)
  - obs[14]: body_y_velocity，身体世界y方向速度（横向速度），reward_usable: true (可用，但非主要)
  - obs[15]: body_z_velocity，身体垂直速度，reward_usable: true (可用于平稳性)
  - obs[16]: body_roll_velocity，滚转角速度，reward_usable: true (姿态稳定性)
  - obs[17]: body_pitch_velocity，俯仰角速度，reward_usable: true
  - obs[18]: body_yaw_velocity，偏航角速度，reward_usable: true (非必须，但可约束)
  - obs[19]: joint_1_velocity，第一个髋关节角速度，reward_usable: false (很少用)
  - obs[20]: joint_2_velocity，第一个踝关节角速度，reward_usable: false
  - obs[21]: joint_3_velocity，第二个髋关节角速度，reward_usable: false
  - obs[22]: joint_4_velocity，第二个踝关节角速度，reward_usable: false
  - obs[23]: joint_5_velocity，第三个髋关节角速度，reward_usable: false
  - obs[24]: joint_6_velocity，第三个踝关节角速度，reward_usable: false
  - obs[25]: joint_7_velocity，第四个髋关节角速度，reward_usable: false
  - obs[26]: joint_8_velocity，第四个踝关节角速度，reward_usable: false

注意：前进方向对应obs[13]，且info完全不可访问，所有必须的奖励信号只能来自obs和action。

## 4. 动作空间 action_space
- type: Box
- shape: [8]
- dtype: float
- 连续动作空间，每个分量范围[-1.0, 1.0]，代表关节力矩。
  - action_dim 0: hip_1_torque，第一个髋关节力矩
  - action_dim 1: ankle_1_torque，第一个踝关节力矩
  - action_dim 2: hip_2_torque，第二个髋关节力矩
  - action_dim 3: ankle_2_torque，第二个踝关节力矩
  - action_dim 4: hip_3_torque，第三个髋关节力矩
  - action_dim 5: ankle_3_torque，第三个踝关节力矩
  - action_dim 6: hip_4_torque，第四个髋关节力矩
  - action_dim 7: ankle_4_torque，第四个踝关节力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，只有达到时间上限truncated才算“存活满时间”。
- failure-like termination:
  - body_height_outside_healthy_range：高度低于0.2或高于1.0（跌倒/弹飞），视为失败。
  - state_value_outside_finite_range：任何状态值变为NaN或inf，视为失败。
- ambiguous termination: 无
- truncation: time_limit_reached，表示达到最大步数，并非失败，应该视为中性或轻微正向。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false (只能通过terminated推断，但官方不允许直接使用terminated标记作为奖励信号，除非我们从状态判断)
- allowed_info_fields: []  (info为空)
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 等全部被禁止

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段（此处info为空，禁止使用任何info字段）
- training_progress 只有prompt明确允许时才用（本次未获允许，所以不能用）

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片
- 环境内部未暴露的任何信号（如接触力、关节力矩等）

## 7. 可用于奖励函数的信号
- position: body_z (obs[0]), quaternion (obs[1:5]) 可计算直立投影 (body_up_z = 1 - 2*(quat_x²+quat_y²))
- velocity: body_x_velocity (obs[13]), body_y_velocity (obs[14]), body_z_velocity (obs[15]), body_roll_velocity (obs[16]), body_pitch_velocity (obs[17]), body_yaw_velocity (obs[18])
- orientation: quat可直接用于惩罚翻滚/俯仰过大
- contact: 无
- action/engine: action 力矩本身可用于平滑性惩罚
- other: 身体高度是否在健康区间内 (可根据obs[0]判定濒死区域)

## 8. 不确定或不可用的信号
- 绝对位置x,y：不可用（被禁止）
- 接触力/脚掌压力：观察空间中没有
- 是否有脚掌离地、步态事件等：没有
- 成功/失败标志：无，只能从高度判跌倒
- 训练进度training_progress: 未允许使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D_quadruped_rigid_body
  actuator_type: torque_controlled_joints (8 DoF: 4 hip + 4 ankle)
  contact_structure: foot_ground_contacts (implicit, not observable)
primary_objectives:
  - maximize_forward_velocity (body_x_velocity)
  - maintain_healthy_body_height (0.2 < body_z < 1.0)
  - keep_body_upright (body_up_z close to 1.0)
secondary_objectives:
  - penalize_large_joint_torques (action)
  - penalize_lateral_drift (body_y_velocity)
  - penalize_high_angular_velocities (roll/pitch/yaw)
main_failure_risks:
  - falling over (body_z < 0.2)
  - launching or jumping too high (body_z > 1.0)
  - NaN/Inf instability (poorly tuned actions)
  - getting stuck with zero forward velocity but still standing
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_reward
  purpose: 激励向前移动的速度
  why_required: 任务的核心目标是前进，缺少此责则无法形成运动策略
  usable_signals: [obs[13] (body_x_velocity)]
  risks: 仅激励速度可能忽略稳定性，需与生存条件结合

- role_id: body_height_survival
  purpose: 驱使机器人保持身体高度在健康区间内，远离上下边界
  why_required: 健康高度是继续任务的必要条件，一旦超出回合终止，因此需设计为惩罚接近边界
  usable_signals: [obs[0] (body_z)]
  risks: 仅仅使用区间内奖励可能导致不思进取，需要设计成边界处急剧下降的势能

- role_id: upright_orientation_penalty
  purpose: 维持机身大致直立，防止翻滚
  why_required: 过度倾斜会导致无法有效向前移动，且容易跌倒
  usable_signals: [obs[1:5] quaternion 可计算 body_up_z]
  risks: 过强惩罚可能抑制必要的侧向倾斜，影响转弯或平衡调整

### 10.2 条件职责 conditional_roles
- role_id: joint_torque_regularization
  condition_to_use: 当策略已经能稳定前进后，用于降低能耗、提高动作效率
  usable_signals: [action (力矩)]
  risks: 早期加入过强的力矩惩罚会阻碍探索有效步态

- role_id: lateral_velocity_penalty
  condition_to_use: 当策略出现明显的横向漂移且影响前进效率时启用
  usable_signals: [obs[14] (body_y_velocity)]
  risks: 轻微侧移可能在某些步态下是自然现象，过度压制可能导致步态僵硬

- role_id: vertical_velocity_penalty
  condition_to_use: 当机器人出现过度跳跃或上下颠簸时启用
  usable_signals: [obs[15] (body_z_velocity)]
  risks: 机器人在行走时垂直方向速度本就不为零，阈值需仔细设计

- role_id: angular_velocity_penalty
  condition_to_use: 当姿态变化过于剧烈时启用，辅助保持稳定
  usable_signals: [obs[16], obs[17], obs[18]]
  risks: 抑制转弯灵活性，不建议在训练早期使用

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_bonus_based_on_time_survived
  reason: 没有info字段可用来获得生存步数，且training_progress不可用；强行从环境隐含状态推断复杂且不可靠，且可能误导reward shaping
  forbidden_or_missing_signals: [无准确生存时间信号]

- role_id: foot_contact_coordination
  reason: 环境未提供任何接触力或触地检测信号
  forbidden_or_missing_signals: [contact sensors]

- role_id: distance_from_origin
  reason: 官方明确禁止使用x_position或distance_from_origin，且这些信号不直接从 obs 暴露
  forbidden_or_missing_signals: [x_position, y_position]

- role_id: curriculum_on_velocity_target
  reason: 不允许使用官方reward项，且无法获取训练进度变量
  forbidden_or_missing_signals: [training_progress, any curriculum signal]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_reward | obs[13] (body_x_velocity) | 无 | dense_state_signal → clip or linear scale | 可直接乘以正系数，或使用有上限的饱和函数防止异常大值 |
| body_height_survival | obs[0] (body_z) | 无 | bounded_signal → penalty for near_boundaries (outside healthy range) | 设计为健康区域为0代价，接近边界急剧上升的二次或指数惩罚 |
| upright_orientation_penalty | obs[1:5] → body_up_z = 1 - 2*(q_x^2+q_y^2) | 无 | bounded_signal → 1 - body_up_z 作为惩罚 | body_up_z在[0,1]，直立为1 |
| joint_torque_regularization | action[0:8] | 无 | quadratic_penalty on action | 需缩放权重以免与前进奖励竞争 |
| lateral_velocity_penalty | obs[14] | 无 | absolute_value or squared penalty | 谨慎权重，可能为0或很小 |
| vertical_velocity_penalty | obs[15] | 无 | absolute_value or squared penalty, with small deadzone | 行走自然会有小幅上下波动 |
| angular_velocity_penalty | obs[16], obs[17], obs[18] | 无 | sum of squares | 对roll/pitch特别敏感，yaw可忽略 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 策略收敛到静止站立不动（零前进速度） | body_x_velocity始终接近0，身体高度稳定 | 检查高度生存奖励是否过强压制了前进项，降低生存奖励权重或增加速度敏感度 |
| 频繁跌倒或过早终止（高度过低） | 平均 episode 长度短，body_z经常接近0.2 | 加强高度惩罚的陡峭程度，或在奖励中增加平滑的“健康高度”势能 |
| 机器人以怪异姿势前进（如倒立或侧翻行走） | body_up_z 显著小于1，但身体高度仍在范围内且前进速度快 | 提高直立惩罚系数，或重新训练 |
| 高 oscilation / 关节抖动 | action 变化剧烈，joint_velocities 很大 | 加入 torque_regularization 和平滑惩罚 |
| 前进方向漂移严重 | body_y_velocity 绝对值很大 | 加入横向速度惩罚 |
| NaN/Inf 终止频繁 | 日志中出现非有限值 | 检查动作输出是否有极大值，clip 或降低力矩范围，或加入平稳状态奖励 |



# expert_reward_context.md

# Expert Schema Context（非检索版）

这份内容不是 RAG 检索结果，也不是按 benchmark 名称写死的奖励模板。它是给 Reward Generator 使用的固定专家 Schema：先读 environment_card.md 中的任务画像和奖励职责拆解，再从下面的小型公式算子库中选择合适数学形式。

核心顺序必须是：

```text
环境事实 → 任务画像 → 奖励职责 reward roles → 职责-信号映射 → 公式算子 → reward code
```

不要反过来先套某个 skeleton 名称。模板只提供专家思考方式，不构成封闭候选集合。

---

## 1. Expert Schema 使用规则

- environment_card.md 中的 `expert_task_profile`、`reward_role_decomposition`、`role_to_signal_mapping` 优先级最高。
- 本文件只提供通用公式算子，不替代环境卡片。
- 先选 role，再选 signal，再选 formula operator，最后写 compute_reward。
- 如果某个 role 需要的信号不可用，必须排除，不得硬写。
- 如果任务画像与模板不完全一致，以 environment_card.md 的可用信号和禁止信号为准。
- 不要因为模板中出现某个 role，就机械加入该 role。
- reward_v1 优先覆盖主学习信号和必要健康约束；效率、能耗、复杂门控和动态权重默认留到后续迭代。

---

## 2. Formula Operator Library

每个算子包含：数学形式、适用场景、触发证据、反模式。

### 2.1 dense_state_signal
- 适用职责：持续前进、速度、姿态、高度、接近目标等连续状态职责。
- 常见形式：
  - positive (线性): `w * signal`
  - positive (凸化): `w * signal**2` 或 `w * exp_form`
    凸化形式在 signal 较大时提供更强梯度。触发证据：episode 长度正常但 score 停滞在低水平，且该信号的 episode_sum_mean 始终偏小——说明 agent 满足于低水平稳态，需要凸化奖励来打破。
  - penalty (二次): `-w * error**2`
  - penalty (hinge): `-w * max(0, threshold - signal)` 或 `-w * max(0, signal - upper)`
    hinge 只在超出安全区间时生效，避免在安全范围内持续惩罚正常波动。触发证据：约束组件的 active_rate≈100% 但 terminated 率仍然很高——说明"全时惩罚"没有给 agent 安全探索空间，它无论怎么调整都被罚。
- 使用条件：该状态信号每步可观测，且与任务目标直接相关。
- 风险：线性正奖励可能导致慢速平台；凸化形式若权重过大可能诱导极端行为；hinge 的 threshold 设太宽则防护不足。

### 2.2 bounded_signal
- 适用职责：限制速度、距离、姿态误差或其他连续信号的极端值。
- 常见形式：
  - 平滑压缩: `x / (1 + abs(x))`
  - 倒数衰减: `1 / (1 + k * abs(error))`
  - 线性衰减: `max(0, 1 - abs(error) / threshold)`
- 使用条件：原始信号可能过大、尺度不稳定，或信号容易被刷分。
- 触发证据：某个信号的 episode_sum_mean 出现极端值（远大于其他组件），说明无界形式被 exploit。
- 风险：threshold 过小会导致反馈饱和或无梯度。
- 反模式：不要用 bounded_signal 替代 hinge penalty——如果目标是"只在越界时惩罚"，用 dense_state_signal 的 hinge 形式，不要用 bounded 包围。

### 2.3 improvement_delta
- 适用职责：接近目标、距离减少、状态改善。
- 常见形式：
  - `old_measure - new_measure`
  - `next_value - current_value`
- 使用条件：obs 和 next_obs 中存在可比较的当前量与下一步量。
- 触发证据：有明确的目标度量（如到目标的距离）且该度量在 episode 中单调递减时 agent 表现好。
- 风险：目标附近可能震荡；没有明确目标度量时不要使用。
- 反模式：不要对速度类信号用 improvement_delta——持续速度本身已经是"进步"，delta 会退化为噪声。

### 2.4 potential_based_shaping
- 适用职责：有明确 potential function 的任务塑形。
- 常见形式：`gamma * Phi(next_obs) - Phi(obs)`
- 使用条件：能够从环境信号定义合理的 Phi。
- 风险：错误 Phi 会误导策略；reward_v1 不默认使用，除非任务天然适合。

### 2.5 quadratic_penalty
- 适用职责：姿态误差、角速度、动作幅度、速度等轻量约束。
- 常见形式：`-w * error**2` 或 `-w * sum(action_i**2)`
- 使用条件：约束信号可观测，且不应压制主学习信号。
- 风险：权重过大会导致 agent_afraid_to_move 或 over_conservative_policy。
- 触发证据：某维度出现高频大幅波动或极端值，但没有触发终止——说明需要轻量抑制而非硬约束。
- 反模式：不要对"有明确安全边界"的信号用 quadratic_penalty（如身体高度必须在 0.2-1.0）。quadratic 从中心开始罚，会让 agent 困在中心不敢动；应改用 hinge 形式只在边界附近生效。

### 2.6 soft_health_gate
- 适用职责：让主进展奖励在健康状态下充分生效，而不是直接加大惩罚。
- 常见形式：`main_reward * gate_factor`，gate_factor 在身体状态恶化时从 1 平滑衰减到 0。
  - 倒数门: `1 / (1 + k * abs(posture_error))`
  - 线性衰减门: `max(0, min(1, (signal - danger) / margin))`
- 使用条件：terminated 主要由健康/安全违规导致，且主奖励在失败回合中仍然显著为正。
- 触发证据（关键）：terminated 率高（>50%）且主进展信号在失败回合的 episode_sum 仍然 >0——说明 agent 在"先冲后死"，需要 gate 在健康恶化时切断主奖励，而不是加一个独立惩罚。
- 风险：gate 太严格会抑制探索；gate 的衰减区间应设在"接近危险但尚未终止"的范围内。
- 反模式：不要用"加大独立惩罚系数"替代 gate。如果 terminated 是因为身体状态越界，单纯加大该状态的惩罚（Level 1）通常不如将其作为 gate 乘到主奖励上（Level 2），因为惩罚只在越界后才生效，gate 在越界前就开始衰减主信号。

### 2.7 joint_condition_proxy
- 适用职责：多个条件必须同时满足的软完成近似，例如 near + low speed + stable。
- 常见形式：`factor_1 * factor_2 * factor_3`，每个 factor 都是连续 bounded 形式。
- 使用条件：没有显式 success flag，但有连续信号可构造 soft proxy。
- 触发证据：agent 能在各个子条件上分别取得进展，但无法同时满足——说明缺一个"联合满足"的引导信号。
- 风险：乘积容易塌缩（一个 factor 趋近 0 则整体为 0）；使用 `(factor_1 + factor_2 + ...) / n` 或几何平均 `(factor_1 * factor_2 * ...) ** (1/n)` 可缓解。
- 反模式：不要用二值条件做乘积——每个 factor 必须是连续函数，否则乘积退化为稀疏信号。

### 2.8 curriculum_weighting
- 适用职责：早期探索和后期精细控制明显冲突时。
- 常见形式：`early_weight = 1 - training_progress`，`late_weight = training_progress`
- 使用条件：training_progress 明确允许，且确有阶段性需求。
- 风险：增加消融混杂；reward_v1 默认不要使用。

---

## 3. 迭代修改时的算子切换指南

以下映射帮助 reflection agent 从"训练反馈证据"直接定位到"该选哪个算子做 Level 2 变换"。
不要求组件名完全匹配；以数学语义和训练表现证据为准。

| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |


```
