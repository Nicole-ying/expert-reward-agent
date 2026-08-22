# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器着陆任务。飞行器从画面顶部中心附近以初始随机推力开始运动。核心目标是 **尽可能快且稳定地降落在中心目标垫上**（位置归零、速度很低、姿态竖直、双腿安全接触地面）。次要目标是 **减少引擎燃料消耗**。智能体必须学习靠近目标点、减速、保持竖直、安全触地，同时避免坠毁、飞出屏幕或不稳定着陆。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: `high`  
reason: 任务的核心是到达指定的固定目标垫（位置和姿态确定），能耗和速度是次要优化项。符合 goal-reaching 定义，不具有存活式平衡、驾驶安全、多目标冲突等特征。

dynamics_subtype: `goal_approach_and_soft_contact`  
说明：飞行器需要接近目标、减速、姿态归正，并让两条支撑腿与垫子发生安全低碰撞接触，属于典型的“接近+软接触”动力学。  

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: 推测为 `float32`
- 各维度均可用作奖励信号（reward_usable: true）

| 索引 | 名称                  | 含义                              | reward_usable |
|------|-----------------------|-----------------------------------|---------------|
| 0    | x_position            | 水平坐标，相对于目标垫中心         | true          |
| 1    | y_position            | 垂直坐标，相对于目标垫高度         | true          |
| 2    | x_velocity            | 水平方向线速度                     | true          |
| 3    | y_velocity            | 垂直方向线速度                     | true          |
| 4    | body_angle            | 机体倾角                           | true          |
| 5    | angular_velocity      | 角速度                             | true          |
| 6    | left_support_contact  | 左支撑腿是否接触垫子（1.0/0.0）    | true          |
| 7    | right_support_contact | 右支撑腿是否接触垫子（1.0/0.0）   | true          |

## 4. 动作空间 action_space
- type: `Discrete`
- n: `4`
- 动作含义：

| 动作值 | 名称                      | 含义                                     |
|--------|---------------------------|------------------------------------------|
| 0      | no_engine                 | 无推力，仅受重力与惯性                   |
| 1      | left_orientation_engine   | 点燃左侧姿态引擎（产生逆时针旋转力矩）     |
| 2      | main_engine               | 点燃主引擎（产生向上的推力）               |
| 3      | right_orientation_engine  | 点燃右侧姿态引擎（产生顺时针旋转力矩）     |

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：身体与地面/障碍物发生严重碰撞（坠毁），一般为失败。
- **horizontal_position_outside_viewport**：水平位置超出视口边界，失败。
- **body_not_awake_or_settled**：身体进入休眠或稳定停止。这既可能是因为成功着陆稳定，也可能因坠毁后静止，属于**模糊终止**。
- **无主动截断（truncation）**（任务环境未明确定义时间限制）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: `false`
- explicit_failure_flag_available: `false`
- allowed_info_fields: `{}` （info 为空字典）
- forbidden_or_uncertain_info_fields: 任何 info 内的字段均不可用

> 智能体无法直接通过信息字典获知成功/失败；必须从观测状态（尤其是位置、速度、接触标志）与终止标志的组合推断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    ...
```

**允许使用**：
- `obs`、`action`、`next_obs` 的全部内容
- `info` 中由本卡片明确声明为可用的字段（当前为**无**）
- `training_progress` 仅当需要时可用这里没有要求，暂不使用

**严格禁止**：
- `original_reward`（官方奖励已被隐藏）
- 任何未在本卡片第3节/第5.2节中声明的 `obs` 额外切片或 `info` 字段

## 7. 可用于奖励函数的信号
- **位置**：`x_position`（希望接近 0）、`y_position`（希望接近 0，表示在垫子上）
- **速度**：`x_velocity`、`y_velocity`（着陆时希望接近 0）
- **角度**：`body_angle`（着陆时希望接近 0，即竖直）
- **角速度**：`angular_velocity`（希望接近 0）
- **接触**：`left_support_contact`、`right_support_contact`（双撑脚同时为 1 表示安全着陆）
- **动作**：离散动作可为惩罚省燃料的依据（0 为无推力，1/2/3 为使用某种引擎）

## 8. 不确定或不可用的信号
- 任何 `info` 中的字段：不存在，无法使用
- `original_reward`：被屏蔽，禁止参考
- 显式的成功/失败标志：无
- 任务完成后的稀疏 terminal bonus：无（因 compute_reward 中不传入 `done`）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D_lander
  actuator_type: discrete_thrusters (main, left/right_orientation)
  contact_structure: two_support_legs
primary_objectives:
  - reach_target_pad (x->0, y->0)
  - stable_soft_landing (low velocity, upright angle, both legs touching)
secondary_objectives:
  - minimal_engine_usage (fuel efficiency)
  - fast_settlement (time-to-land)
main_failure_risks:
  - crash_into_ground_or_obstacle
  - fly_out_of_viewport
  - excessive_oscillation (rollover)
  - landing_on_one_leg_or_with_large_tilt
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `approach_and_landing_shaping`  
  **purpose**: 提供连续稠密信号，引导飞行器靠近目标垫、减速、保持竖直并触发双腿接触。  
  **why_required**: 任务成功的全部物理条件（位置、速度、姿态、双撑脚接触）均需通过此信号塑造，否则智能体无法学到安全着陆。  
  **usable_signals**:  
    - `x_position`, `y_position`（距离目标远近）  
    - `x_velocity`, `y_velocity`（控制着陆冲击）  
    - `body_angle`, `angular_velocity`（保持竖直）  
    - `left_support_contact`, `right_support_contact`（安全着陆确认）  
  **risks**: 若各分量权重不平衡，可能导致智能体只顾位置而忽略姿态，或过度减速导致悬停过久。

### 10.2 条件职责 conditional_roles
- **role_id**: `action_penalty`  
  **purpose**: 惩罚非零引擎动作，以鼓励节省燃料。  
  **condition_to_use**: 始终可加入，但惩罚幅度不能大到阻碍必要的飞行控制。  
  **usable_signals**: 离散动作（0 为无推力，1-3 为使用引擎）  
  **risks**: 过强惩罚会导致智能体不敢点火，无法抵消重力、飞出边界；过弱则对燃料节省无效。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `terminal_success_bonus`  
  **reason**: 奖励函数接口不提供 `done` 或 `terminated` 标志，无法仅在一局结束时发放一次性的成功奖励。可用稠密 shaping 替代。  
  **forbidden_or_missing_signals**: done 标志  
- **role_id**: `info_based_failure_penalty`  
  **reason**: `info` 为空字典，不包含任何失败指标。  
  **forbidden_or_missing_signals**: info 字段

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_and_landing_shaping | x_position, y_position, x_velocity, y_velocity, body_angle, angular_velocity, left_support_contact, right_support_contact | （无） | distance_to_target, bounded_signal, quadratic_penalty, contact_bonus (when y near 0) | 核心奖励，所有信号均可用 |
| action_penalty | action (discrete 0/1/2/3) | （无） | discrete_action_cost (e.g. 0 for no thrust, penalty for others) | 配合使用需控制强度 |
| terminal_success_bonus | （无） | done/terminated flag | — | 禁用 |
| info_based_failure_penalty | （无） | info failure field | — | 禁用 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 自由落体撞击 | y_velocity 很大且 y_position 快速减小，最终 crash | 增加靠近垫子时的速度惩罚，强化减速需求 |
| 飞出水平视口 | abs(x_position) 持续增大，最终终止 | 增加 x 偏离惩罚，或 reward x 向中心的吸引 |
| 姿态失控翻转 | abs(body_angle) 或 angular_velocity 剧增 | 加大对角度和角速度的惩罚，使用二次或绝对值 |
| 悬停犹豫不决 | episode 过长，y_position 徘徊在中高度 | 引入时间惩罚或降低悬停奖励，甚至给予负奖励促进下降 |
| 单腿着陆失败 | 最终只有一只脚接触，另一只悬空或撞击 | 在 y 接近 0 时显著奖励双腿接触，惩罚仅单腿接触 |
| 过度使用引擎 | 连续非零动作，燃料消耗高 | 适度增强 action_penalty，或按使用频率衰减奖励 |