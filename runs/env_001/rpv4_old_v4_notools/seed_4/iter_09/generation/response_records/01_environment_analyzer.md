# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 车辆/着陆器轨迹优化问题。主体从顶部中心附近开始，带有初始随机力。主要目标是**到达并稳定停靠在中央目标垫上**（即位置归零、速度接近零、姿态稳定且双腿安全接触）。次要目标是**尽可能快**地完成该过程，同时**尽量少用引擎推力**（节约燃料）。  
不要将该任务与纯粹的燃料最小化问题混淆，因为到达目标仍是首要约束。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 任务核心是让主体到达固定目标区域并稳定下来，终态条件基于位置、速度、接触状态，属于典型的导航到达任务；快速和低燃耗只是性能品味，不构成单独冲突目标。

动力学子类型 dynamics_subtype: **goal_approach_and_soft_contact**  
理由：主体必须在接近目标时主动减速，调整姿态，并用两条支撑腿安全接触目标垫，符合“接近 → 减速 → 稳定接触”的动力学模式。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32 (默认)  
- **obs[0]**: `x_position` — 相对于目标垫的水平坐标，reward_usable: true  
- **obs[1]**: `y_position` — 相对于目标垫高度的高度坐标，reward_usable: true  
- **obs[2]**: `x_velocity` — 水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity` — 垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle` — 机体倾斜角，reward_usable: true  
- **obs[5]**: `angular_velocity` — 角速度，reward_usable: true  
- **obs[6]**: `left_support_contact` — 左支撑腿接触标志（0/1），reward_usable: true  
- **obs[7]**: `right_support_contact` — 右支撑腿接触标志（0/1），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- **action 0**: `no_engine` — 不点火，零推力  
- **action 1**: `left_orientation_engine` — 点燃左侧姿态引擎  
- **action 2**: `main_engine` — 点燃主引擎（通常向上推力）  
- **action 3**: `right_orientation_engine` — 点燃右侧姿态引擎  

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 当主体稳定在目标垫上（位置≈0，速度≈0，双腿接触）时触发，极大概率为成功。
- **failure-like termination**:  
  - `crash_or_body_contact` — 任何意外的身体碰撞（如翻倒、腿以外的部分触地）导致失败。  
  - `horizontal_position_outside_viewport` — 水平位置超出视口，丢失目标区域。
- **ambiguous termination**: `body_not_awake_or_settled` 也可能出现在失败坠落并静止于非法区域（如地面远处），但其纹理通常包含速度近零、躺倒状态；但按环境定义，该条件大概率用作成功稳定着陆。
- **truncation**: 代码中无显式步数截断（`terminated` 是组合布尔，无 `Truncated` 返回），因此无自动截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**  
- explicit_failure_flag_available: **false**  
- allowed_info_fields: 无（info 为空字典 `{}`）  
- forbidden_or_uncertain_info_fields: 任何 info 字段（包括 `success`、`failure`、`termination_reason` 等）均不存在，**禁止使用**。

## 6. reward 函数接口契约
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs` — 当前步观察（8维数组）
- `action` — 执行的动作（0~3）
- `next_obs` — 下一步观察
- `info` — 只允许读取已声明的字段，当前为空，因此**禁止依赖 info**
- `training_progress` — 仅在 prompt 明确允许时使用，**当前不可用**

禁止使用：
- `original_reward` — 官方奖励被掩盖，禁止直接或间接使用
- `official_reward` 变体
- 任何未声明的 obs 切片（如非公开的全局状态）
- 任何未经声明的 info 字段

## 7. 可用于奖励函数的信号
- position:  
  - `x_position`（obs[0]）, `y_position`（obs[1]），均为相对于目标垫的坐标，可用于计算距离、接近程度、高度惩罚等。  
- velocity:  
  - `x_velocity`（obs[2]）, `y_velocity`（obs[3]），可用于速度阻尼、冲击惩罚。  
- orientation:  
  - `body_angle`（obs[4]）, `angular_velocity`（obs[5]），可用于姿态稳定惩罚。  
- contact:  
  - `left_support_contact`（obs[6]）, `right_support_contact`（obs[7]），0/1值，可用于鼓励或惩罚接触状态。  
- action/engine:  
  - `action` 本身，可用于燃料消耗惩罚（非零动作视为引擎使用）。  
- other: 无。

## 8. 不确定或不可用的信号
- 绝对位置或速度（仅有相对目标值）  
- 时间步数、剩余步数（没有在 obs 或 info 中暴露）  
- 燃料剩余量或具体推力值  
- 任何成功/失败标签  
- 外部干扰（如风）的显式测量

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 着陆器（刚体）
  actuator_type: 三个离散推力器：主引擎、左姿态引擎、右姿态引擎
  contact_structure: 两条支撑腿，通过左/右接触标志感知接触
primary_objectives:
  - 将位置 (x,y) 移动到目标垫原点附近（距离趋于0）
  - 在目标垫上以极低速度稳定着陆（vx≈0, vy≈0）
  - 完成着陆时双腿均接触目标垫，姿态接近竖直（angle≈0）且角速度≈0
secondary_objectives:
  - 最小化引擎动作次数（燃料效率）
  - 尽快终止（隐性目标，但无可直接奖励的时间信号）
main_failure_risks:
  - 机体翻倒或非腿部接触（crash）
  - 水平偏离目标垫太远，飞出视口
  - 高速撞击目标垫（穿模/弹起未稳）
  - 在目标垫附近无限期悬停（浪费燃料但未终止）
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: `goal_proximity`  
  purpose: 将主体拉向目标位置（x→0, y→0）。  
  why_required: 这是到达目标的核心驱动力，否则主体无动力向目标移动。  
  usable_signals: [x_position, y_position]  
  risks: 若权重过高可能驱使高速撞击；需与速度阻尼配合。

- **role_id**: `velocity_damping`  
  purpose: 在接近目标时抑制过高的线速度，防止撞击或弹起。  
  why_required: 仅目标接近不足以诱导稳定着陆，需要减速。  
  usable_signals: [x_velocity, y_velocity]  
  risks: 过强阻尼会使主体停滞在高空，阻碍探索；需与位置奖励平衡或仅在距离较近时激活。

- **role_id**: `orientation_stabilization`  
  purpose: 惩罚大倾角和高角速度，鼓励保持竖直姿态。  
  why_required: 最终着陆需要双腿平稳接触，姿态偏差会导致翻滚或单腿着地失败。  
  usable_signals: [body_angle, angular_velocity]  
  risks: 早期阶段过度惩罚角运动可能抑制必要的姿态调整。

- **role_id**: `soft_contact_encouragement`  
  purpose: 鼓励双腿同时接触目标垫，且接触时速度足够小以避免撞击惩罚。  
  why_required: 最终终止条件要求稳定的双腿接触，仅靠位置、速度奖励可能双腿虚接或单腿着陆。  
  usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity]  
  risks: 直接奖励接触可能导致主体匆忙“摔”向垫子；需结合低速条件。

### 10.2 条件职责 conditional_roles
- **role_id**: `fuel_efficiency_penalty`  
  condition_to_use: 当主体已较为接近目标垫（例如距离 < 阈值）时激活，以避免在远程探索时过多惩罚必要的推力。  
  usable_signals: [action]  
  risks: 若全程惩罚非零动作，会严重阻碍探索，使主体干脆不使用引擎而飘离。

- **role_id**: `final_settlement_bonus`  
  condition_to_use: 仅在满足全部稳定条件（极低速度、双腿接触、姿态小、位置小）时赋予一次性奖励，否则不给予。  
  usable_signals: [x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact]  
  risks: 单独使用不会产生平滑梯度，仅作为终端增强；需配合稠密奖励使用。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_efficiency_reward`  
  reason: 缺少步数计数或剩余时间信号，无法直接奖励快速完成。  
  forbidden_or_missing_signals: [current_step, remaining_steps]

- **role_id**: `external_collision_penalty`  
  reason: 碰撞事件由环境直接终止轨迹，无法在终止帧计算奖励（终止时不会调用奖励函数或需特殊处理），且 info 无碰撞标记。  
  forbidden_or_missing_signals: [collision_flag]

- **role_id**: `official_reward_imitation`  
  reason: 官房奖励被屏蔽，不得重建或模仿。

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | — | negative_distance, quadratic_penalty, bounded_potential | 推荐用距离的负值或反比，避免因二次项导致远处梯度过大 |
| velocity_damping | x_velocity, y_velocity | — | velocity_magnitude_penalty, squared_speed, soft_clip | 可以与距离门控（如仅当距离<阈值时应用）结合 |
| orientation_stabilization | body_angle, angular_velocity | — | absolute_angle_penalty, squared_penalty | 角度和角速度可分离加权 |
| soft_contact_encouragement | left_support_contact, right_support_contact, x_velocity, y_velocity | — | contact_product * speed_penalty_inverse, conditional_bonus | 鼓励双腿接触且速度小；可设计为近距离下的激励 |
| fuel_efficiency_penalty | action | engine_force_info (不可用) | action_nonzero_indicator, step_penalty | 仅在近距离激活，避免阻碍初期探索 |
| final_settlement_bonus | x_position, y_position, x_velocity, y_velocity, body_angle, left_support_contact, right_support_contact | — | all_condition_satisfied_bonus, sparse_terminal_reward | 一次性奖励，置放于所有指标小于阈值时 |

## 12. 初始训练后应观察的 failure modes

| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体从不接近目标，飘出视口 | 奖励曲线不上升，位置偏差持续很大 | 检查目标 proximity 奖励权重是否过小，或速度阻尼是否过早限制推力 |
| 高速撞击目标垫并弹出/翻倒 | 接触前速度很大，轨迹终止于 crash | 增强临近垫时的速度惩罚，或引入与距离成反比的“下降速度限制”奖励 |
| 长时间在高空悬停，不下降 | y 位置保持较大正值，速度接近零 | 增加高度惩罚或渐进式高度减小奖励，或减少动作惩罚以鼓励向下推力 |
| 单腿着陆无法终止 | 只有一条腿接触，主体微小摇摆但无法稳定 | 加强 soft_contact 奖励对双腿同时接触的激励，同时微调姿态稳定权重 |
| 过度使用姿态引擎导致颠簸 | 动作频繁为 1 或 3，角速度震荡 | 在靠近垫子时激活 fuel_efficiency_penalty，减轻姿态震荡 |
