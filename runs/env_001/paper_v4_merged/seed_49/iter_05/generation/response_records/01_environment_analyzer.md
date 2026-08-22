# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 轨迹优化任务，控制一个飞行/滑行器从视口顶部中央出发，快速到达并平稳停靠在中央目标垫上。核心目标是准确到达目标位置并安全、稳定地接触目标垫；次目标是尽量缩短时间、减少引擎推力使用。agent 必须学会接近目标、主动减速、保持正确姿态（接近水平）、安全软着陆。严禁只为省油而长时间悬停，也禁止过度推力导致坠毁或飞出视野。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 环境明确要求“reach and settle at a central target pad”，存在固定的目标位置，并且接近、减速、姿态控制均服务于到达目标并稳定停靠这一核心目标。省推力仅为附属优化，不构成等权对立的多目标，因此选为导航目标到达。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32  
各维度含义：
- obs[0]: x_position — 相对于目标垫中心的水平坐标（左负右正），reward_usable: true
- obs[1]: y_position — 相对于目标垫高度的垂直坐标（上正下负），reward_usable: true
- obs[2]: x_velocity — 水平线速度，reward_usable: true
- obs[3]: y_velocity — 垂直线速度，reward_usable: true
- obs[4]: body_angle — 机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity — 机体角速度，reward_usable: true
- obs[6]: left_support_contact — 左侧支撑腿是否接触（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact — 右侧支撑腿是否接触（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，仅靠惯性运动
- action 1: left_orientation_engine — 点燃左侧姿态引擎（产生绕质心的力矩）
- action 2: main_engine — 点燃主引擎（产生沿机体纵轴方向的推力，方向由 body_angle 决定）
- action 3: right_orientation_engine — 点燃右侧姿态引擎（反向力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
三个终止条件（任一触发即 `terminated=True`）：
- crash_or_body_contact — 坠毁或身体接触（具体定义被掩码，但结合任务说明，非支撑腿的“身体接触”通常指本体/其它部位触碰地面，属于失败）
- horizontal_position_outside_viewport — 水平位置超出视口边界
- body_not_awake_or_settled — 机体不再活跃（awake）或已稳定停靠（settled）  
  需注意：settled 可能对应成功着陆（速度极小、接触良好），但也可能意味着机体因翻倒/卡住而进入静止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，无显式成功字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为 {}，但后续若出现也不应使用，除非 prompt 明确允许）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不允许直接使用；尤其禁止假设存在 success、failure、termination_reason 等字段。

**间接推断路径（derived_possible）**  
成功的终端状态可能表现为：左右支撑腿同时接触（obs[6] 和 obs[7] 均接近 1.0）、位置靠近目标（|obs[0]| 小，|obs[1]| 小）、速度极低、角度接近 0。失败则可能表现为：出界（终止前最后一步 obs[0] 已大幅偏离）、机体角度过大、垂直速度过大导致硬冲击（derived_possible）。由于终止时无法获得明确的标注，奖励函数若使用终端事件，只能基于观测信号间接推断，并标注 derived_possible。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
- 允许使用：obs, action, next_obs, 以及 info 中 prompt 明确允许的字段（当前为空，无允许字段）
- 禁止使用：original_reward（official_reward 被掩码，不可访问），任何未声明的 info 字段，任何未在 observation_space 中声明的 obs 切片。
- training_progress 只有在 prompt 明确允许时才可使用（当前环境未提及），默认禁止。

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，可构造距离 (√(x²+y²))
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，可构造合速度 magnitude
- orientation: obs[4] (body_angle)，obs[5] (angular_velocity)
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact)，可推断安全着陆
- action/engine: action 选择 (0,1,2,3) 可用于计算推力使用
- other: 通过 next_obs 与 obs 的差值可派生 delta_distance、delta_velocity 等变化量；终止前最后几步的观测序列（若可用）可用于推导成功/失败事件（derived_possible）

## 8. 不确定或不可用的信号
- 无显式的 success/failure 标签
- 无燃料、能耗的直接测量值（只能通过 action 选用主引擎次数间接估计）
- 无目标坐标的显式 distance-to-goal 或 reward shaping 信号（官方 reward 被掩码）
- 无接触力、冲击力信息

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 轻质飞行器/着陆器，具有两个支撑腿用于安全着陆检测
  actuator_type: 离散动作：主推力引擎（沿机体轴）、左右姿态引擎（产生力矩）
  contact_structure: 仅左右支撑腿有接触传感器，本体接触地面则可能触发 crash
primary_objectives:
  - 到达并稳定停靠在目标垫上（x,y 同时趋近于 0，且速度极低，姿态平坦）
  - 通过左右支撑腿同时接触实现安全着陆
secondary_objectives:
  - 最小化耗时（尽快到达，但不牺牲安全着陆）
  - 最小化引擎推力使用（鼓励惯性滑行，但不得因节油而悬停）
main_failure_risks:
  - 过度使用主引擎导致高速撞击目标垫（硬着陆）
  - 姿态控制不当使机体侧翻或倾斜着陆（单腿触地或本体触地）
  - 水平方向飘移出视野
  - 为省油而长期悬停在目标上方却不着陆（悬停陷阱）
  - 过早减速导致的漫长缓慢逼近（效率低下）
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- role_id: delta_distance_to_target
  purpose: 鼓励每一步向目标垫靠近，形成通往目标区域的稠密梯度。实现为连续两步之间到目标点距离的减少量 (delta)。  
  why_required: 目标位置的观测坐标已给出，且任务核心是“到达”。使用 delta 而非 proximity 可避免悬停收割（停在靠近但不着陆的区域仍能获得正分）。  
  usable_signals: [obs[0], obs[1]] (前一步与下一步均可获得)  
  risks: 在接近目标后，梯度变弱，需要终端事件或终端成形进行补偿；若距离计算未归一化，奖励量级可能过大。

- role_id: terminal_success_bonus (derived_possible)
  purpose: 在安全着陆发生时给予一次性正奖励，强化完成整个任务的行为。  
  why_required: 让 agent 明确知道最终稳定着陆是值得的，弥补后期稠密梯度不足。  
  usable_signals: [left_support_contact, right_support_contact, x_position, y_position, x_velocity, y_velocity, body_angle]  —— 当 episode 终止且左右支撑均接触、位置与速度均在小阈值内时推断成功。  
  risks: 若推断阈值过于严苛，可能导致奖励过于稀疏；若过松，可能误将部分失败（如翻倒后刚好两腿触地）也判为成功。需标注 derived_possible。

### 10.2 条件职责 conditional_roles
- role_id: energy_penalty
  condition_to_use: 当且仅当任务描述明确要求“尽可能少用引擎推力”时加入；当前描述明确提及“using as little engine thrust as possible”，因此有条件加入。  
  usable_signals: [action] — 当 action=2 (主引擎) 时给予小惩罚；姿态引擎（1，3）也可酌情小罚，但主引擎推力大，惩罚更重。  
  risks: 权重过大可能导致 agent 完全不使用推力而无法到达目标；可以用非常小的系数，或只在主引擎连续多个步骤使用时加重惩罚。

- role_id: survival_health_gate
  condition_to_use: 若主奖励使用 forward/velocity 相关信号，但在 agent 倾覆或失控后该信号不再可靠时启用。但本环境主信号是 delta_distance，该信号即使在倾覆时也可能保持有效（距离仍可能减少但状态不好），因此可选性较低；但可作为一种安全门，当姿态严重倾斜时对主奖励进行抑制。  
  usable_signals: [body_angle] — 当 |angle| 大于阈值时对 delta_distance 奖励进行衰减或归零。  
  risks: 若阈值设置不当可能抑制正常的姿态调整（转向时略微倾斜），仅应在观察到 agent 经常翻转训练时才启用。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: proximity_reward (距离的单调函数，如 -distance)
  reason: 会鼓励 agent 停在靠近但不着陆的位置持续得分（悬停）。在未加入强制着陆机制前禁用。  
  forbidden_or_missing_signals: 无，但数学形态不宜单独作为主奖励。

- role_id: velocity_penalty (无条件的速度惩罚)
  reason: 会与 “快速到达” 目标冲突；减速应是接近目标后的行为，不应全程惩罚。若需要减速，使用 conditional velocity penalty 仅在靠近目标时生效。  
  forbidden_or_missing_signals: 无，但用法不合目标。

- role_id: hard_contact_penalty (仅凭接触标志加罚)
  reason: 接触标志只能表示支撑腿触地，无法区分软着陆和硬冲击。不能以此作为失败惩罚，易造成错误惩罚安全的着陆。  
  forbidden_or_missing_signals: 缺失冲击力传感器，无法可靠区分。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| delta_distance_to_target | x_position, y_position (obs和next_obs) | - | delta, l2_distance, improvement | 采用 next_distance - current_distance 的负值或 (current - next) 作为密度奖励 |
| terminal_success_bonus | left_support_contact, right_support_contact, x_position, y_position, x_velocity, y_velocity, body_angle (derived_possible) | 显式 success flag | sparse_event_bonus, threshold_gate | 推断成功条件：(both_contacts) & (distance<eps) & (speed<eps) & (|angle|<eps) |
| energy_penalty | action | 燃料传感器 | penalty_on_action (action==main_engine) 小系数 | 避免惩罚太重导致不推进行为；可仅对连续主引擎动作累加微罚 |
| survival_health_gate | body_angle | - | gate (1 if |angle|<thresh else 0) | 仅作为 delta_distance 的乘性门控 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停陷阱：agent 在目标正上方 y≈0, x≈0 但不着陆，持续微小推力抵消重力 | 训练的 episode length 极高，reward 曲线停滞但未达成功终端；平均 y_velocity 接近 0 且 contact flags 保持 0 | 增加 delta_distance 奖励在距离很小时的衰减，或加入时间惩罚 / terminal success 增强；降低 energy penalty 权重确保 agent 有动力完成着陆 |
| 硬着陆/撞击后反弹：高速冲向地面，短暂着地后 episode 结束或终止 | 监视 terminal 前几步的 velocity magnitude 过大，contact flags 变为 1 又瞬间结束 | 添加 conditional velocity damping 奖励：在接近地面（y 很小）时奖励低速度；或在推断成功时引入速度门控，高速时仅给低额成功奖励 |
| 翻转或一侧着陆后卡住：body_angle 大幅偏离 0，仅单腿 contact | 单侧 contact_flag 为 1，另一侧为 0，且 body_angle 绝对值大 | 引入姿态惩罚（hinge），仅在 |angle|>阈值时生效；同时 survival gate 抑制主奖励 |
| 漂移出界：水平位置超出视口 | 观测 x_position 在一连串步骤中持续增大（或减小）直至 episode 突然终止 | 添加 out-of-bounds soft penalty，根据 |x| 在接近边界时施加递增惩罚（带 hinge） |
| 推力使用极端：从未使用主引擎（依赖初始速度）导致无法到达或长时间飘荡 | action=2 频率极低，episode 长度大且距离收敛慢 | 适度提高 energy penalty 的上限，或将其改为“超出允许推力预算才罚”的形式，避免压抑必需推力 |
