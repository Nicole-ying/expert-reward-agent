# 匿名环境理解卡片

## 1. 任务目标
主目标：控制一个具有主引擎和左右方向引擎的2D飞行器，从起始区域（大致上方正中）开始，尽可能快速地降落到视野中央的着陆平板上，并实现稳定停靠（姿态平稳、速度接近于零、双脚接触标志均为真）。  
次目标：在完成主目标的前提下，尽可能减小引擎推力使用量（即降低燃料消耗）。  
不可混淆的目标：单纯存活、无目的悬停或长时间飞行都不符合任务意图，高油耗不应作为唯一优化方向。

## 2. 任务类型选择
- selected_route_id: **navigation_goal_reaching**
- confidence: **high**
- reason: 核心目标是到达指定空间位置（中央着陆平板）并安全停靠，属于典型的“导航目标到达”任务。省燃料、快速到达等附属目标不影响主分类。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: [8]
- dtype: float32
- 各维度含义：

| 索引 | 字段名 | 含义 | reward_usable |
|------|--------|------|---------------|
| 0 | x_position | 飞行器相对着陆平板中心的水平坐标 | true |
| 1 | y_position | 飞行器相对着陆平板顶部高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 飞行器倾斜角度（弧度） | true |
| 5 | angular_velocity | 角速度 | true |
| 6 | left_support_contact | 左脚接触标志（1.0 接触，0.0 未接触） | true |
| 7 | right_support_contact | 右脚接触标志（1.0 接触，0.0 未接触） | true |

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作编号 | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 无任何引擎点火，自由运动 |
| 1 | left_orientation_engine | 点火左侧方向引擎（可能产生逆时针力矩或左向推力） |
| 2 | main_engine | 点火主引擎（通常产生向上的推力） |
| 3 | right_orientation_engine | 点火右侧方向引擎（可能产生顺时针力矩或右向推力） |

说明：左右方向引擎用于调整飞行器姿态，主引擎主要用于减速/悬停，具体物理效果由掩码动力学决定。

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` —— 飞行器稳定停靠（通常意味着降落在平板上且速度、角速度都趋于零）。此时双脚接触标志很可能都为1。
- **failure-like termination**:  
  - `crash_or_body_contact` —— 身体非正常碰撞（例如与地板强烈撞击、侧翻导致其他部位接触），可能包含未落在平板上的情况。  
  - `horizontal_position_outside_viewport` —— 水平位置超出视野边界，失控飞出。
- **ambiguous termination**: 若 `body_not_awake_or_settled` 发生在非平板区域（理论上不会，因为接触标志可能为0），但目前信息无法完全排除。
- **truncation**: 作业描述中未提及 episode 截断步数，但通常 RL 环境存在默认步数上限，此处未声明，暂不考虑。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: **false**（info 返回空字典）
- **explicit_failure_flag_available**: **false**
- allowed_info_fields: {}（无可用字段）
- forbidden_or_uncertain_info_fields: 任何 info 内的字段均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

**允许使用**：
- `obs`（完整 8 维向量，含义见上）
- `action`（0~3 整数）
- `next_obs`（下一时刻的 8 维向量）
- `info` 中明确声明可用的字段（目前无任何字段可用）
- `training_progress` 仅当 prompt 明确允许时可用（本任务并未允许，因此**禁止使用**）

**严格禁止**：
- `original_reward`（官方奖励已被掩码，不可参考）
- 任何未声明的 `info` 字段（如 `info['success']` 等）
- 任何未列出的 `obs` 切片假设

## 7. 可用于奖励函数的信号
从 `obs` 和 `next_obs` 中可提取：
- **位置**：x_position, y_position（相对平板中心的高度差）
- **速度**：x_velocity, y_velocity
- **姿态**：body_angle, angular_velocity
- **接触**：left_support_contact, right_support_contact（0或1）
- **动作/推力**：action 编号（可间接计算是否使用了引擎）
- **派生量**：可计算到目标的水平距离 sqrt(x_pos² + y_pos²)，速度范数，是否双脚全部接触（contact_all = left*right），是否接近静止等。

## 8. 不确定或不可用的信号
- 终止原因无法从 info 获取，reward 函数内无法获知 episode 是否已终止（除非额外传入 terminated 标志，但当前接口不包含）
- 无法区分“crash”与“settle”终止信号，只能通过状态（接触、速度、角度）事后推断
- 无法获取步长限制信息
- 无法获取当前环境真实几何尺寸（如平板宽度、高度限制），依靠经验调节阈值

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D_lander (主引擎 + 两侧方向姿态引擎)
  actuator_type: thruster (离散开关)
  contact_structure: two_legs_with_contact_indicators
primary_objectives:
  - 准确降落到中央着陆平板中心
  - 实现双脚稳定接触并静止 (速度和角速度趋于零)
secondary_objectives:
  - 最小化引擎使用次数（降低燃料消耗）
  - 缩短降落飞行时间（隐含快速性）
main_failure_risks:
  - 水平飞出视野边界
  - 剧烈碰撞导致 crash（可能因高速或侧翻）
  - 在平板外区域静止（非成功停靠）
  - 过度使用引擎导致无法收敛到节能策略
```

## 10. 奖励职责拆解 reward_role_decomposition

### 10.1 主职责 mandatory_roles
- **role_id**: approach_to_target
  - purpose: 鼓励飞行器不断缩小与着陆平板中心的距离
  - why_required: 必须驱动 agent 向目标位置移动，是任务最基础的前进方向
  - usable_signals: [x_position, y_position] （可直接构造 距离惩罚 = sqrt(x² + y²) 或分段）
  - risks: 距离惩罚过重可能导致高速冲撞，需与减速职责配合

- **role_id**: soft_landing_velocity_control
  - purpose: 当飞行器接近平板时迫使速度减小，实现软着陆
  - why_required: 单纯接近目标可能导致高速撞击，必须抑制速度
  - usable_signals: [x_velocity, y_velocity]
  - risks: 阈值设置不当会抑制正常下降，或产生过多控制震荡

- **role_id**: stable_contact_termination_bonus
  - purpose: 当检测到双脚均接触、速度/角速度接近于零时，给予较大的成功奖励
  - why_required: 提供最终目标完成的稀疏强化信号，否则不易学会精确停靠
  - usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, angular_velocity]
  - risks: 如果奖励过于稀疏可能学习缓慢，如果阈值过于宽松可能误判

- **role_id**: orientation_stabilization
  - purpose: 在接近至平板附近时，惩罚过大倾斜角度和角速度，防止侧翻 crash
  - why_required: 姿态失控会导致快速失败，必须在降落阶段维持平衡
  - usable_signals: [body_angle, angular_velocity]
  - risks: 远离平板时不必强制姿态修正，须加入距离条件以免过度约束

- **role_id**: fuel_efficiency_penalty
  - purpose: 对主引擎和方向引擎的使用进行轻微惩罚，促使节省燃料
  - why_required: 任务明确要求“使用尽可能小的引擎推力”，不加入会学到滥推
  - usable_signals: [action]（非零动作即表示点火）
  - risks: 惩罚权重过高会导致 agent 不愿点火，完全放弃控制；需组合其他职责平衡

### 10.2 条件职责 conditional_roles
- **role_id**: speed_modulation_in_freefall
  - condition_to_use: 当飞行器距离平板较远（下降初期），可对水平速度持续惩罚以维持大致垂直下落，但非必选，可由主职责覆盖
  - usable_signals: [x_velocity]
  - risks: 可能抑制必要的横向移动

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: survival_time_reward
  - reason: 任务目标是尽快降落到平板，而非长时间存活，无生存奖励信息支持
  - forbidden_or_missing_signals: [episode_step, alive_flag]
- **role_id**: exploration_bonus
  - reason: 状态空间连续且目标明确，不需要刻意探索奖励
- **role_id**: multi_objective_energy_distance_balance
  - reason: 虽有多指标，但主次分明，无需多目标权衡框架

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_to_target | `x_position`, `y_position` | none | `euclidean_distance`, `negative_distance`, `quadratic_penalty` | 距离越近奖励越大 |
| soft_landing_velocity_control | `x_velocity`, `y_velocity` | none | `velocity_norm`, `quadratic_penalty`, `distance_thresholded_scale` | 只有靠近平板时才有效 |
| stable_contact_termination_bonus | `left_contact`, `right_contact`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity` | none | `all_contacts_true`, `norm_small`, `step_function_bonus` | 条件：双脚接触 and 速度角速度极小 |
| orientation_stabilization | `body_angle`, `angular_velocity` | none | `absolute_value`, `quadratic_penalty`, `distance_gated` | 加以距离门控避免过早约束 |
| fuel_efficiency_penalty | `action` | none | `indicator_nonzero` | 惩罚任何非零引擎动作 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 水平飞出视野边界 | episode 提前终止，x_position 绝对值持续增大 | 加强横向速度惩罚，或增加目标方向引导奖励 |
| 高速撞击平板导致 crash | 接触时速度很大，双脚接触可能为1但 reward 很低（因 crash终止） | 施加速度惩罚并提高软着陆奖励的阈值灵敏度 |
| 在平板边缘悬停或滑动但不停靠 | 接触标志部分为1，速度不为零，episode 不停（被截断） | 增加成功接触的严格条件（双脚接触+极低速度），提高成功奖励密度 |
| 过度翻滚（body_angle 较大）后 crash | 角速度或角度持续增大 | 增加姿态稳定惩罚的权重或提前触发条件 |
| 燃料消耗过大（不停点火） | 每个动作点火的频率极高，但可能由于缺乏惩罚而不学节能 | 启动燃料效率惩罚，但注意平衡以避免不动作导致任务失败 |