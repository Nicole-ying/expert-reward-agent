# Response Record

# 匿名环境理解卡片

## 1. 任务目标
任务主目标：控制一个二维飞行器从顶部中心区域出发，快速、平稳地降落到画面中央的目标着陆垫上，并在着陆瞬间保持稳定的姿态和速度，使左右支撑腿都与垫面接触。次目标：在满足成功着陆的前提下，尽可能缩短到达时间和节省发动机推力（燃料）。不应混淆为目标只是悬停或单纯到达上空，必须实际软着陆并稳定停靠。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 核心目的是到达并稳定停留在指定目标点（着陆垫），属于“导航到目标”任务族。附属目标（快速、省燃料、姿态稳定）是为了提升着陆质量，不改变任务主目标，因此不属于多目标任务。动力学子类型选为 goal_approach_and_soft_contact，因为要求接近目标、减速、控制姿态并安全接触。

## 3. 观察空间 observation_space
- type: Box
- shape: [8]
- dtype: float32 (推断)
- obs[0]: x_position，飞行器相对于目标着陆垫的水平坐标，reward_usable: true
- obs[1]: y_position，飞行器相对于着陆垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线性速度，reward_usable: true
- obs[3]: y_velocity，垂直线性速度，reward_usable: true
- obs[4]: body_angle，机体朝向角度，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0.0 或 1.0），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0.0 或 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不激活任何发动机，无推力输出
- action 1: left_orientation_engine，激活左侧姿态调整发动机（产生角加速度和可能的侧向力）
- action 2: main_engine，激活主发动机（产生向上的推力，同时可能影响姿态）
- action 3: right_orientation_engine，激活右侧姿态调整发动机（与左姿态发动机对称）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机体静止/稳定），如果此时 x,y 位置非常接近目标、速度近乎零且左右接触标志均为 1.0，则大概率是成功着陆。
- failure-like termination: crash_or_body_contact 和 horizontal_position_outside_viewport 明确对应坠毁、接触不良或飞出视口。
- ambiguous termination: body_not_awake_or_settled 也可能发生在未到达目标但外界原因导致静止的情况，需要结合位置和接触进一步判断。
- truncation: 未明确给出，但可能存在最大步数截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，没有任何可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均被禁止使用，包括假想的 "success"、"failure"、"termination_reason"

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- obs：当前观察向量（8维）
- action：当前执行的动作（0~3）
- next_obs：执行动作后的下一观察向量（8维）
- info：空字典 {}，无可用字段
- training_progress：暂时无需使用（prompt 未明确要求）

禁止使用：
- original_reward（被屏蔽的内部奖励）
- 任何未在 obs/next_obs 中明确声明的信号
- 任何 info 字段

## 7. 可用于奖励函数的信号
- position: 
  - `obs[0]` (x_position, 水平偏差)
  - `obs[1]` (y_position, 垂直偏差)
- velocity: 
  - `obs[2]` (x_velocity)
  - `obs[3]` (y_velocity)
- orientation: 
  - `obs[4]` (body_angle)
  - `obs[5]` (angular_velocity)
- contact: 
  - `obs[6]` (left_support_contact)
  - `obs[7]` (right_support_contact)
- action/engine: 
  - `action` 本身（0~3），可区分不同发动机使用，用于计算燃料代价
- other: 
  - 从上述信号可构造综合接近程度、着陆稳定性、姿态水平等派生信号

## 8. 不确定或不可用的信号
- 目标位置不是观测的一部分，而是通过 obs[0] 和 obs[1] 的相对值体现；无法直接获得目标的绝对坐标或是否达到目标区域的 bool 标志。
- 没有显式的 "success" 或 "landed" 布尔信号。
- 没有燃料剩余量或时间步数的直接信息（除非环境提供，但这里没有）。
- 没有外部 reward 或 info 中的阶段指示。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 双足/双支撑着陆飞行器 (lander-like)
  actuator_type: 主推力发动机 + 两个姿态调整发动机
  contact_structure: 左右两个着陆支撑腿，接触时产生 1.0 标志
primary_objectives:
  - 到达并稳定停留在目标着陆垫（位置偏差极小，速度近零）
  - 着陆时同时产生左右支撑腿接触
secondary_objectives:
  - 尽可能短的时间到达（间接通过步数惩罚实现）
  - 节省燃料（减少发动机使用）
main_failure_risks:
  - 坠毁或机体部分异常接触（crash_or_body_contact）
  - 水平飞出视口边界
  - 过早“稳定”但未在目标区域（误触发 body_not_awake_or_settled）
  - 着陆时姿态过倾斜导致单腿接触或翻倒
  - 过度使用主发动机导致燃料耗尽或剧烈震荡
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_target
  purpose: 驱动飞行器向目标点移动，缩小水平与垂直偏差。
  why_required: 到达目标是主任务，必须提供持续的趋近信号。
  usable_signals: [obs[0], obs[1]] （位置偏差）
  risks: 仅用距离可能导致高速冲撞，需要配合减速。

- role_id: soft_landing_velocity
  purpose: 在接近目标时降低速度，实现软着陆。
  why_required: 着陆必须速度近零，否则即使位置正确也会失败或坠毁。
  usable_signals: [obs[2], obs[3]] （线速度），可与位置距离联合塑造（速度惩罚随距离衰减）
  risks: 全局惩罚速度会抑制初期快速接近，需要逐步激活或距离门控。

- role_id: upright_stabilization
  purpose: 保持机体姿态水平，避免倾斜过度。
  why_required: 着陆时姿态稳定是成功接触的前提，左右接触要求姿态接近零。
  usable_signals: [obs[4], obs[5]] （角度、角速度）
  risks: 过度惩罚轻微摆动可能导致保守控制，可结合即将着陆的信号加强。

- role_id: successful_contact_reward
  purpose: 奖励两腿同时接触的状态。
  why_required: 最终着陆成功的硬性指标，无此信号无法判断任务完成。
  usable_signals: [obs[6], obs[7]] （接触标志乘积或和）
  risks: 仅靠接触奖励可能导致飞行器提前摆出接触姿势而不真正到达目标，必须与位置和速度联合使用。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency
  purpose: 惩罚不必要的发动机使用，节省燃料。
  condition_to_use: 在整个 episode 中持续生效，但权重不宜过高以免干扰主目标学习。
  usable_signals: [action] （通过区分无动作、主引擎、姿态引擎赋予不同惩罚）
  risks: 早期训练可能因惩罚导致 agent 不敢使用推力，可采用线性衰减或很小系数。

- role_id: time_penalty
  purpose: 鼓励尽快完成任务。
  condition_to_use: 每步给予小的负奖励（活代价），或在成功着陆时给予一次性的与步数成反比的奖励。
  usable_signals: 隐式，可通过在 episode 结束时测量步数实现，但 compute_reward 单步接口无法直接获取步数；可以用每步小的固定负奖励模拟。
  risks: 固定步惩罚过大可能使 agent 急功近利而坠毁。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: orientation_engine_usage_penalty_alone (单独惩罚姿态发动机)
  reason: 姿态发动机是必要控制手段，一味惩罚会阻止 agent 学习稳定姿态，应与姿态误差配合使用。
  forbidden_or_missing_signals: 无可直接判断是否“浪费”姿态发动机的信号，仅靠动作类型无法区分有效调整与无意义乱喷。

- role_id: exact_contact_sequence_bonus
  reason: 缺少阶段标签或脚力传感器，无法推断正确的着陆顺序（如先左后右），强行设计可能导致脆弱奖励。
  forbidden_or_missing_signals: 无接触时序信息，仅有当前帧的 bool 标志。

- role_id: success_flag_based_reward
  reason: 环境未提供显式的 success 标志，info 为空，根本不可用。
  forbidden_or_missing_signals: info 中无任何字段。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | obs[0], obs[1] | 无 | dense_state_signal (基于距离函数，如 -sqrt(x^2+y^2) 或 shaped exponential) | 距离最小化是基础 |
| soft_landing_velocity | obs[2], obs[3] | 无 | bounded_signal (可结合距离门控，如 -||v|| * f(dist) ) | 防止高速撞击 |
| upright_stabilization | obs[4], obs[5] | 无 | quadratic_penalty (角度平方 + 角速度平方) | 姿态保持水平 |
| successful_contact_reward | obs[6], obs[7] | 无 | logical_and_or_sum ( reward when both are 1, maybe only when also near target ) | 着陆成功标志 |
| fuel_efficiency | action | 无 | discrete_action_penalty (不同 action 赋予不同负值) | 节省燃料 |
| time_penalty | 无直接信号 | 真实步数 | fixed_constant_penalty (每步 -small_value) | 加速完成 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器悬停在空中不下降 | y_position 长期为正，y_velocity 约等于零，主发动机使用频繁 | 降低燃料惩罚权重或增加下降推力引导信号，如对负 y_velocity 给予奖励 |
| 高速砸向目标然后反弹或坠毁 | 终止前速度范数极大，终止时 crash_or_body_contact 触发 | 加强软着陆速度惩罚，尤其是低高度时；可以引入速度上限惩罚 |
| 到达目标但姿态大幅倾斜，单腿接触 | obs[4] 绝对值大，且 obs[6] + obs[7] < 2.0 时终止 | 增加姿态稳定权重，或在接近目标时放大姿态误差的惩罚 |
| 长时间左右摆动，无法稳定 | angular_velocity 持续振荡，接触不断变化 | 对角速度施加平滑惩罚，或结合速度/姿态联合塑造更稳定的下降动力学 |
| 一味节省燃料而不用主发动机，导致缓慢漂离 | 距离增加，y_velocity 缓慢正向，很少使用 action 2 | 提高主任务趋近奖励的权重，或者允许适度燃料使用，或动态调整燃料惩罚系数 |
| 在目标附近稳定但未触发终止（可能被 truncation 截断） | episode 结束但 position error 很小，接触标志不为全1 | 检查环境终止条件是否对“稳定”判定过于严格；可考虑在接近目标时略微增大接触奖励，激励 agent 完成最后压触动作 |
