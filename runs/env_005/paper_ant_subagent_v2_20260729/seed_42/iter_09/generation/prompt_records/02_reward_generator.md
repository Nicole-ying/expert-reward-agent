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
控制一个 3D 四足机器人（8 个扭矩控制关节）实现稳定前进。核心目标是保持机器人身体直立且高度位于健康范围（0.2~1.0），并产生持续的向前速度。次要目标是提升前进速度，但不得以牺牲稳定性为代价。任务不是简单的“存活/保持平衡”，而是在存活前提下最大化前进进度。若机器人倾倒或高度越界，episode 提前终止。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 任务主目标是在保持平衡（健康高度、直立）的条件下持续前进，属于典型的“持续前进通过地形”类控制，到达目标位置或抓取操作不可见。速度是附属优化，前进进度才是主干，因此不属于 survival_balance_task（并非只求不倒）或多目标冲突型。

## 3. 观察空间 observation_space
- type: Box
- shape: [27]
- dtype: float (continuous)
- obs[0]: body_z, 机身垂直高度, reward_usable: true（健康门控、摔倒推断）
- obs[1]: quat_w, 身体朝向四元数实部, reward_usable: true（与 quat_x,y,z 联合用于直立计算）
- obs[2]: quat_x, 身体朝向四元数虚部 x, reward_usable: true
- obs[3]: quat_y, 身体朝向四元数虚部 y, reward_usable: true
- obs[4]: quat_z, 身体朝向四元数虚部 z, reward_usable: true
- obs[5]: joint_1_angle, hip_1 角度, reward_usable: true（关节状态/姿态偏好）
- obs[6]: joint_2_angle, ankle_1 角度, reward_usable: true
- obs[7]: joint_3_angle, hip_2 角度, reward_usable: true
- obs[8]: joint_4_angle, ankle_2 角度, reward_usable: true
- obs[9]: joint_5_angle, hip_3 角度, reward_usable: true
- obs[10]: joint_6_angle, ankle_3 角度, reward_usable: true
- obs[11]: joint_7_angle, hip_4 角度, reward_usable: true
- obs[12]: joint_8_angle, ankle_4 角度, reward_usable: true
- obs[13]: body_x_velocity, 机身世界 x 轴前进速度, reward_usable: true（主进度信号）
- obs[14]: body_y_velocity, 机身世界 y 轴侧向速度, reward_usable: true（侧向漂移惩罚）
- obs[15]: body_z_velocity, 机身垂直速度, reward_usable: true（与健康/摔倒相关）
- obs[16]: body_roll_velocity, 翻滚角速度, reward_usable: true
- obs[17]: body_pitch_velocity, 俯仰角速度, reward_usable: true
- obs[18]: body_yaw_velocity, 偏航角速度, reward_usable: true
- obs[19]: joint_1_velocity, hip_1 角速度, reward_usable: true（动作平滑/能耗）
- obs[20]: joint_2_velocity, ankle_1 角速度, reward_usable: true
- obs[21]: joint_3_velocity, hip_2 角速度, reward_usable: true
- obs[22]: joint_4_velocity, ankle_2 角速度, reward_usable: true
- obs[23]: joint_5_velocity, hip_3 角速度, reward_usable: true
- obs[24]: joint_6_velocity, ankle_3 角速度, reward_usable: true
- obs[25]: joint_7_velocity, hip_4 角速度, reward_usable: true
- obs[26]: joint_8_velocity, ankle_4 角速度, reward_usable: true

## 4. 动作空间 action_space
- type: Box (continuous)
- shape: [8]
- action[0]: hip_1_torque, 第一个髋关节扭矩，范围 [-1, 1]
- action[1]: ankle_1_torque, 第一个踝关节扭矩，范围 [-1, 1]
- action[2]: hip_2_torque, 第二个髋关节扭矩，范围 [-1, 1]
- action[3]: ankle_2_torque, 第二个踝关节扭矩，范围 [-1, 1]
- action[4]: hip_3_torque, 第三个髋关节扭矩，范围 [-1, 1]
- action[5]: ankle_3_torque, 第三个踝关节扭矩，范围 [-1, 1]
- action[6]: hip_4_torque, 第四个髋关节扭矩，范围 [-1, 1]
- action[7]: ankle_4_torque, 第四个踝关节扭矩，范围 [-1, 1]

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止（如到达目标）。但环境中唯一正常的“存活到截断”可视为软成功，因为 agent 未因失败而终止。
- failure-like termination: body_height_outside_healthy_range（高度 <0.2 或 >1.0，很可能指示摔倒/崩溃或跳起）；state_value_outside_finite_range（数值发散/崩溃）。
- ambiguous termination: 无。
- truncation: time_limit_reached（步数耗尽），属于正常截断，不与任何具体失败绑定。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 为空字典，禁止任何 info 字段）
- forbidden_or_uncertain_info_fields: reward_forward, reward_ctrl, reward_contact, reward_survive, x_position, y_position, distance_from_origin 均禁止使用。info 中无 termination_reason 标志。
- 推断路径（derived_possible）：
  - **失败（摔倒/崩溃）**：可从 body_z（obs[0]）突然低于 0.2 或 body_z_velocity 突变、身体 quat 指示大幅度倾斜（body_up_z 远小于 1）等组合中检测。但这些信号在终止时间步可能仍在过渡中，可靠性中等。
  - **截断为“存活到时间结束”**：若 episode 被 truncated 且摔倒信号未触发，则可推断为存活完成。
  - **前进任务的“成功”**：不能被单步信号直接判断，只能在 episode 结束后根据存活时间长度与正向速度累积来推断，但无法在 reward 的 step 计算内访问 episode 全局。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前观测，但不要在步奖励中依赖未来或历史状态）
- action
- next_obs
- training_progress（仅用于退火或 schedule，如果训练框架允许，但谨慎使用以避免环境特定依赖）

禁止使用：
- original_reward（被掩码的官方奖励）
- 任何 info 字段（即使环境中存在也不允许）
- x_position / y_position 或任何全局位置信号
- official_reward 的任何成分

## 7. 可用于奖励函数的信号
- position: body_z (obs[0]) – 机体高度
- orientation: quat_w,x,y,z (obs[1:5]) – 可从四元数计算 body_up_z ≈ 1 – 2*(x²+y²)，衡量机体直立程度
- velocity (body): body_x_velocity (obs[13]) – 世界 x 轴前进速度（主进度信号）；body_y_velocity (obs[14]) – 侧向速度（可用于惩罚侧滑）；body_z_velocity (obs[15]) – 垂直速度（与摔倒/健康相关）；roll/pitch/yaw angular velocity (obs[16:19]) – 角速度稳定性信号
- velocity (joint): joint_1_velocity ~ joint_8_velocity (obs[19:27]) – 关节角速度
- joint angles: joint_1_angle ~ joint_8_angle (obs[5:13]) – 关节位置
- action/engine: 动作扭矩 action[0:8]，可用于动作幅度平滑惩罚
- other: 通过 body_z 和 quat 组合可间接推断摔倒/健康；通过 body_up_z 可推断倾斜惩罚；通过 body_x_velocity 可反映前进进度

## 8. 不确定或不可用的信号
- 显式成功/失败标志：不可用
- 全局 x 坐标：不可用，无法知道绝对前进距离，也无法直接奖励“完成一段行程”
- 接触力/接触标志：不可用（版本无接触信息）
- 能量/燃料消耗：没有直接功率或扭矩平方和信号，但可通过 action 幅度间接近似
- 来自 info 的任何项：完全禁止
- 速度命令/目标速度：未提供，只能观察实际速度

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: multi_legged_body_locomotion
control_type: continuous
morphology:
  body_type: 3D quadruped with rigid body and four legs
  actuator_type: torque control (8 joints: 4 hips + 4 ankles)
  contact_structure: implicit ground contact via MuJoCo; no explicit contact sensors
primary_objectives:
  - maintain body height in (0.2, 1.0)
  - maintain upright body orientation (body_up_z close to 1.0)
  - produce sustained forward velocity (body_x_velocity > 0)
secondary_objectives:
  - maximize forward speed
  - penalize excessive lateral / vertical velocity (drift, bobbing)
  - smooth joint actuation (keep torques moderate)
main_failure_risks:
  - falling below height threshold (collapse)
  - exceeding height threshold (launch/unstable jump)
  - large tilt leading to unrecoverable orientation
  - numerical instability leading to NaN/Inf state
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_velocity_progress
  purpose: 通过 body_x_velocity 直接反馈前进进度
  why_required: 任务核心是前进，且观测提供明确前进轴速度。该信号连续、密集，是主信号的天然候选。
  usable_signals: [next_obs[13] (body_x_velocity)]
  risks:
    - 若 robot 倾倒但身体仍向前滑动，body_x_velocity 可能维持正数，造成奖励空洞；需与健康门控联合使用。
    - 过早 reward 较大时，agent 可能只追求速度而牺牲稳定性，需正交惩罚介入。

- role_id: healthy_height_gate
  purpose: 根据 body_z 是否在安全范围 (0.2, 1.0) 内，生成一个在 [0,1] 的门控因子，与前进奖励相乘；一旦高度接近边界，进度奖励迅速衰减。
  why_required: 防止 forward_velocity 在非健康状态（尤其是接近跌倒）时仍产生强正反馈，避免 policy 学会“倒前冲一把”或“低爬滑行”。
  usable_signals: [next_obs[0] (body_z)]
  risks:
    - 边界计算需要平滑（如 hinge），否则可能导致数值梯度断裂。
    - 门控太严格会使早期探索无法获得任何奖励，需初始宽容区。

- role_id: upright_penalty
  purpose: 用 body_up_z = 1 – 2*(quat_x²+quat_y²) 衡量的直立程度，当低于阈值时施加 hinge penalty。
  why_required: 方向是保持平衡的关键部分；纯高度门控无法防止机身大幅度俯仰/翻滚，因为躯干高度可能仍在范围内但已大幅倾斜，导致不稳定步态。
  usable_signals: [next_obs[1:5] (quat)]
  risks:
    - 需要使用 hinge，避免在安全倾角范围内无用惩罚干扰步态自然微调。
    - 阈值需经验设定（如 body_up_z < 0.8 时惩罚）。

### 10.2 条件职责 conditional_roles
- role_id: lateral_drift_penalty
  condition_to_use: 如果侧向速度显著或任务希望走直线（task_description 未明确禁止侧向，但“walk or run forward”暗示单向）。
  usable_signals: [next_obs[14] (body_y_velocity)]
  risks:
    - 太强的惩罚可能妨碍转弯的自然步态（但本环境无转弯需求，风险低）。
    - 使用绝对值或平方惩罚时注意幅度缩放。

- role_id: action_smoothness
  condition_to_use: 如果我们期望自然、低抖动的动作，可加入动作变化量惩罚（当前动作与上一步动作之差的范数）。
  usable_signals: [action (8D)]
  risks:
    - 需要在函数内维护上一动作（增加状态需求），或使用 next_obs 的 joint velocity 间接反映。
    - 过量惩罚会降低响应速度和探索能力，权重需轻。

- role_id: joint_velocity_regularization
  condition_to_use: 如果训练早期关节振荡剧烈，可以对 joint_velocity 的 L2 范数进行轻量惩罚。
  usable_signals: [next_obs[19:27]]
  risks:
    - 过度惩罚会抑制步态的大幅摆动，不利于高速行走。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: energy_efficiency
  reason: 没有直接的功率或扭矩平方和与能量读数，只能从 action 幅度推断。但 task_description 未强调“省能”，且早期能量惩罚易与前进探索冲突，因此不作为显式 role，但可通过 action_smoothness 间接轻度贡献。
  forbidden_or_missing_signals: [能量/功率直接测量信号不此版本]
- role_id: target_reaching / distance_to_goal
  reason: 没有目标位置或距离读数；x 坐标不可用。
  forbidden_or_missing_signals: [x_position 禁止使用]
- role_id: foot_contact_reward
  reason: 无 contact 信息，无法判断脚是否触地。
  forbidden_or_missing_signals: [contact flag 不存在]
- role_id: survival_time 作为主信号
  reason: 虽然存活是必要条件，但任务明确要求持续的向前运动，仅存活不足以满足任务。
  usable_signals: 可间接推断存活，但不应作为主进度信号。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_velocity_progress | next_obs[13] (body_x_velocity) | – | dense_state_signal, bounded_signal, linear or log-modulated | 核心进度信号，需与 health gate 相乘 |
| healthy_height_gate | next_obs[0] (body_z) | – | hinge_gate, deadzone_gate | 输出 0~1 gate，接近边界衰减，与 velocity 相乘 |
| upright_penalty | next_obs[1:5] → derived body_up_z | – | hinge_penalty



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





# Fresh Restart Evidence

- target_score: 3000.000
- best_score_so_far: 5.500

## Tried component structures

| structure | attempts | best_score | latest_score | status |
|---|---:|---:|---:|---|
| descent_early_penalty + forward_progress + lateral_drift_hinge + roll_pitch_hinge | 1 | 5.500 | 5.500 | unsolved |
| forward_progress + lateral_drift_hinge + roll_pitch_hinge | 2 | -75.450 | -75.450 | unsolved |
| forward_progress + height_boundary_hinge + lateral_drift_hinge + roll_pitch_hinge | 1 | -186.370 | -186.370 | unsolved |
| action_cost + angular_factor + forward_reward + gate + height_factor + lateral_factor | 1 | -345.870 | -345.870 | unsolved |
| forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty | 2 | -415.210 | -845.860 | unsolved |
| forward_progress + survival_bonus | 1 | -585.810 | -585.810 | unsolved |

## Previous interventions

- iter 2 (score=-75.450, structure=forward_progress + lateral_drift_hinge + roll_pitch_hinge): 4. `selected_level`：Level 2 结构变换，因为需要改变 upright_gate 的核心数学形态（收紧阈值）来消除误报，而非简单尺度调整。 | 5. `selected_intervention`：仅修改 upright_gate，将全开阈值从 0.5 提高到 0.8，零门阈值从 0.2 提高到 0.4，使 forward_progress 只在接近直立（body_up_z > 0.4）时才获得奖励，并且在 body_up_z 在 0.4~0.8 之间线性插值。
- iter 3 (score=-186.370, structure=forward_progress + height_boundary_hinge + lateral_drift_hinge + roll_pitch_hinge): 4. `selected_level`: Level 2 — signal gap for height penalty confirmed by termination rate and gate suppression; add one new component. | 5. `selected_intervention`: Introduce `height_boundary_hinge`: a hinge penalty on body_z outside [0.3, 0.9] with weight 0.8, providing direct negative feedback for height excursions.
- iter 4 (score=5.500, structure=descent_early_penalty + forward_progress + lateral_drift_hinge + roll_pitch_hinge): 4. `selected_level`：Level 2 structure transformation — replace height_boundary_hinge with descent_early_penalty to supply missing predictive signal and remove conflicting standalone penalty. | 5. `selected_intervention`：Remove height_boundary_hinge entirely; add descent_early_penalty that penalizes downward velocity when body_z is below a safe margin (0.5), using obs[15].
- iter 5 (score=-345.870, structure=action_cost + angular_factor + forward_reward + gate + height_factor + lateral_factor): 4. selected_level: Level 3—same skeleton family (forward + independent penalties) has persisted 4 iterations and best score remains <0.2% of target, fulfilling the REBUILD condition. | 5. selected_intervention: Replace all independent constraints with a geometric‑mean health gate (height, upright, lateral, angular including yaw) multiplied by a convex forward‑speed reward (velocity²), plus a tiny actio
- iter 7 (score=-415.210, structure=forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty): 4. selected_level: Level 3 REBUILD，基于历史最优 iter 4 方向（无门控前进 + 轻量运动惩罚），移除生存奖励，引入高度边界惩罚，采用 dense_state_signal 正激励+ hinge 惩罚的主信号骨架。 | 5. selected_intervention: 重新构建奖励为 forward_reward(系数 2.0)、lateral_penalty(hinge,容忍 0.2)、roll/pitch_penalty(hinge,容忍 0.5)、height_low/height_high_penalty(hinge,阈 0.3/0.9)，完全移除 survival_bonus 及任何门控。
- iter 8 (score=-845.860, structure=forward_reward + height_high_penalty + height_low_penalty + lateral_penalty + pitch_penalty + roll_penalty): 4. `selected_level`：Level 2（结构变换）——累积记录显示连续 3 轮未刷新 best，当前“纯线性前进+多项独立惩罚”骨架持续失效；需要在前进奖励中引入 uprightness gate 作为局部反馈，将二值安全条件变为连续的 bounded factor，符合“sparse→dense”模式。 | 5. `selected_intervention`：将 `forward_reward` 组件从 `2.0 * forward_vel` 修改为 `2.0 * forward_vel * uprightness_gate`，其中 `uprightness_gate` 由四元数计算并通过 hinge 函数映射为 [0,1] 连续值；其余惩罚组件保持不变，仅此一个组件变更。

## Restart instruction

The previous search has stagnated. Propose a materially different design hypothesis, not merely a renamed or trivially rescaled copy.
Compare the tried structures and their scores before choosing the next direction.
If you continue a previous structure family, state what new evidence justifies it and change its mathematical mechanism or temporal semantics.
Expert skeletons are design primitives and risk hints, not a closed candidate list. You may combine, transform, or create a new signal using only declared environment inputs.

```
