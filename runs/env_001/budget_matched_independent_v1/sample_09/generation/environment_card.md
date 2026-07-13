# 匿名环境理解卡片

## 1. 任务目标
该环境是一个二维飞行器/着陆器轨迹优化任务。飞行器从视口顶部中心附近释放，并受到随机初始力的扰动。核心目标是**高效、安全地到达目标着陆垫，并在其上稳定停靠**。具体来说，智能体必须学会：
- 快速飞向目标垫（水平与垂直方向同时）
- 在接近时减速，避免猛烈撞击
- 维持稳定姿态，以正确角度与垫面接触
- 以两个支撑腿平稳触地并静止

次要目标是**尽量节省燃料**（减少发动机使用）。该任务不应被理解为单纯的燃料最小化问题，也不等同于保持平衡存活；到达目标才是主任务。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching
confidence: high
reason: 任务描述明确要求“reach and settle at a central target pad”，即导航至目标并稳定停靠，这是典型的到达目标位置任务。虽然存在能耗和速度方面的附属要求，但它们不构成同等权重的多目标冲突，核心评价标准是能否成功着陆。

动力学上进一步细分为 **goal_approach_and_soft_contact**，因为任务不仅要求到达，还要求在接触垫面时姿态稳定、速度较小、且双脚同时或依次安全触地。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float64（推断，实际均为连续值或布尔值转换成的浮点数 0.0/1.0）
- 各维度含义（索引从0开始）：
  - obs[0]: x_position，水平相对目标垫中心的坐标，奖励可用：true（反映水平接近程度）
  - obs[1]: y_position，垂直相对垫面高度的坐标，奖励可用：true（反映高度及着陆进度）
  - obs[2]: x_velocity，水平线速度，奖励可用：true（用于抑制水平晃动或撞击）
  - obs[3]: y_velocity，竖直线速度，奖励可用：true（着陆瞬间的垂直速度关键）
  - obs[4]: body_angle，躯体朝向角度，奖励可用：true（姿态稳定是成功接触的前提）
  - obs[5]: angular_velocity，角速度，奖励可用：true（姿态晃动惩罚）
  - obs[6]: left_support_contact，左支撑腿接触标志（1.0接触，0.0未接触），奖励可用：true（可判定着陆状态）
  - obs[7]: right_support_contact，右支撑腿接触标志，奖励可用：true（同上）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - 动作0：no_engine，不启动任何发动机（惯性滑行）
  - 动作1：left_orientation_engine，启动一个姿态控制发动机（推测产生逆时针或某一方向力矩，用于调整躯体角度）
  - 动作2：main_engine，启动主发动机（推测产生与躯体方向相反的推力，用于减速和上升）
  - 动作3：right_orientation_engine，启动另一个姿态控制发动机（与动作1反向的力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：
  - `body_not_awake_or_settled` ：当身体不再“清醒”（即进入睡眠状态）或已稳定停靠时触发。这通常意味着飞行器已经在目标垫上静止，是成功的着陆迹象。
- **failure-like termination**：
  - `crash_or_body_contact` ：发生碰撞或主体直接接触（可能触地时姿态太差或超出容忍范围）
  - `horizontal_position_outside_viewport` ：水平位置移出视口，偏离着陆区域太远
- **ambiguous termination**：无
- **truncation**：未提供，但环境可能有额外的时间步限制，但 step 源中未体现，故不依赖。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （无明确的 info 字段标明成功）
- explicit_failure_flag_available: false
- allowed_info_fields: [] （info 为空字典 {}，不允许使用任何 info 字段）
- forbidden_or_uncertain_info_fields: 任何 info 字段都不可用，因为 step 返回空字典。

注意：终止条件`body_not_awake_or_settled`虽暗示成功，但它本身是终止标志，并非附加的成功/失败信号。若在奖励函数中依赖该状态，只能通过`done`旗标判断回合结束，而不能区分成功或失败。因此，奖励函数不能直接使用 `info["success"]` 等不存在字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
- **允许使用**：
  - `obs`：当前观测（8维）
  - `action`：当前动作
  - `next_obs`：下一观测（8维）
  - `info`：必须为空字典，不可使用任何键
  - `training_progress`：默认为0.0，仅当提示明确允许时才使用（本次任务未要求），暂不可用
- **禁止使用**：
  - `original_reward` 或任何官方奖励值
  - 未声明的 `info` 字段
  - 未声明的 `obs` 切片或隐含的内部状态

## 7. 可用于奖励函数的信号
- **位置信号**：
  - `x_position`（obs[0]）：可用于衡量水平接近程度
  - `y_position`（obs[1]）：可用于判定是否接近垫面
- **速度信号**：
  - `x_velocity`（obs[2]），`y_velocity`（obs[3]）：着陆瞬间或过程减速
- **姿态信号**：
  - `body_angle`（obs[4]）：姿态稳定与对准
  - `angular_velocity`（obs[5]）：姿态晃动抑制
- **接触信号**：
  - `left_support_contact`（obs[6]），`right_support_contact`（obs[7]）：确认腿部触地，可用于奖励稳定着陆
- **动作/发动机**：
  - 动作编号（0-3），可区分是否使用主发动机或姿态发动机；可用于节约燃料的惩罚

## 8. 不确定或不可用的信号
- 地面真实位置（如目标垫的确切坐标）：不可直接获得，但观测已经转换为相对值，故实际足够。
- 燃料消耗量：未提供，只能通过动作选择间接推断。
- 飞行器质量或惯量：不可用。
- 其他未公开的内部状态（如“awake”标志）：不可直接用于奖励。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body_lander
  actuator_type: discrete_thrusters
  contact_structure: two_point_contact_landing_legs
primary_objectives:
  - 到达目标着陆点（最小化相对位置）
  - 在垫面上稳定停靠（双支撑腿接触且速度接近零）
secondary_objectives:
  - 尽量减少发动机使用次数（节能）
  - 尽快完成着陆（隐含时间效率）
main_failure_risks:
  - 水平漂移出视口
  - 姿态失控导致撞击或侧翻
  - 着陆速度过大造成“crash”
  - 过度使用主发动机飞得太高或再次远离目标
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: approach_and_landing**
  - purpose: 引导飞行器向目标垫移动并最终着陆
  - why_required: 这是任务核心成功标准，必须通过距离减小、速度衰减、姿态对准来塑造。
  - usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]
  - risks: 过分强调距离惩罚可能让智能体高速冲向垫面，必须结合速度惩罚；接触信号不应过早使用，否则会鼓励提前触地而非飞至垫上。

- **role_id: soft_contact_stabilization**
  - purpose: 确保着陆时双支撑腿同时或先后安全触地，且速度、姿态均符合要求
  - why_required: 单腿着陆或翻滚都会导致 `crash_or_body_contact` 失败；稳定触地后需要保持静止以触发 `body_not_awake_or_settled`。
  - usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, angular_velocity]
  - risks: 当飞行器还在空中时，不应因为接触为0而惩罚，只能作为着陆后的正面奖励。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**
  - condition_to_use: 当智能体已经开始接近目标（例如距离已经小于某个阈值）或者可以保证任务成功时，加入动作惩罚以鼓励节省燃料。
  - usable_signals: [action]
  - risks: 过早施加动作惩罚会损害探索，导致智能体什么都不做或不敢使用发动机；必须渐进或仅在任务后期增强。

- **role_id: time_efficiency**（可选）
  - condition_to_use: 若需要鼓励较快到达，可通过给予每个时间步小额负奖励实现；但此环境未提供时间步信息，可以借助恒定的每步惩罚（如与action无关的生存惩罚）。需谨慎使用，避免智能体为快速而撞击垫面。
  - usable_signals: 无独立时间信号，只能以恒定的常数惩罚作为时间驱动力。
  - risks: 可能牺牲安全性。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: precise_posture_alignment**
  - reason: 任务只需要大概的姿态稳定（避免侧翻），不必追求极精确的0度角；过度要求角度可能限制智能体调整姿态的能力。
  - forbidden_or_missing_signals: 无精确姿态目标值，只需在着陆阶段大致垂直，可用但需放松。

- **role_id: exploration_bonus**
  - reason: 任务并非稀疏探索型，距离信号直接提供持续反馈，无需刻意增加好奇心或探索奖励。
  - forbidden_or_missing_signals: 无必要。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_and_landing | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | 无 | distance_penalty, velocity_penalty, angle_penalty, contact_bonus | 距离使用欧几里得或加权；速度在接近地面时加重惩罚；接触作为正面里程碑奖励 |
| soft_contact_stabilization | left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, angular_velocity | 无 | contact_reward (both legs), zero_velocity_bonus, angular_velocity_penalty | 当双腿接触且速度极低时给予高奖励，否则即使接触也极轻微奖励 |
| fuel_efficiency | action | 无 | action_penalty（除no_engine外），或main_engine_penalty | 对动作2（主发动机）可单独增加惩罚，因其耗能最大 |
| time_efficiency | 无专门信号 | time_step | constant_step_penalty | 需谨慎，防止过早终止 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 飞行器过度依赖主发动机，垂直悬浮或飞离目标 | y_position 持续增大或保持较大，而 x 偏离；中期奖励仍为正 | 增大距离惩罚随垂直高度的变化，防止悬停；对持续高耗动作加重惩罚 |
| 水平方向漂移出视口 | 终止于 `horizontal_position_outside_viewport`；训练中水平奖励未改善 | 加强水平位置惩罚，或在视口边缘施加额外斥力信号 |
| 着陆时姿态恶化/侧翻 | 终止于 `crash_or_body_contact`；观察 body_angle 和 angular_velocity 在接近地面时大幅波动 | 在接近垫面且速度较低时，增加姿态角惩罚的权重；对姿态发动机的使用给予宽容（前期不加过大惩罚） |
| 智能体只学习不用发动机，自由落体坠毁 | 低 episode 长度，无接触；奖励多来自无动作惩罚 | 延迟燃料效率奖励，确保任务成功率达标后再引入 |
| 过早单腿触地导致翻滚 | 只有一条腿接触时，身体角速度突然增加后 crash | 在接触奖励中严格要求双腿同时接触，或对单腿接触给予中性奖励而不停步加分 |