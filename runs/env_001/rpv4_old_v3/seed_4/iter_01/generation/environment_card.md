# 匿名环境理解卡片

## 1. 任务目标
这是一个 **2D 飞行器轨迹优化任务**。  
主体在视口上方中央位置出发，受到初始随机力作用。  
**首要目标**：到达画面中央的着陆垫，并稳定停靠（减速、保持姿态、安全接触）。  
**次要目标**：同时尽可能 **快速到达** 且 **节省引擎燃料**。  
**不应混淆的目标**：不是单纯的无限制飞行或持续移动；也不是简单的平衡或抓取任务；核心在于“安全抵达+稳定着陆”。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 核心目标是到达明确的空间位置（着陆垫中心），减速并稳定接触，属于“目标导航到达”任务族。附属的燃料、时间、姿态要求都是在该主目标下的优化约束，不构成并重的多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: `x_position` — 水平相对坐标（相对于目标垫中心），可用于奖励： true
- obs[1]: `y_position` — 垂直相对坐标（相对于着陆垫高度），可用于奖励： true
- obs[2]: `x_velocity` — 水平线速度，可用于奖励： true
- obs[3]: `y_velocity` — 垂直线速度，可用于奖励： true
- obs[4]: `body_angle` — 机体倾斜角度，可用于奖励： true
- obs[5]: `angular_velocity` — 角速度，可用于奖励： true
- obs[6]: `left_support_contact` — 左支撑腿接触标志（1.0 表示接触），可用于奖励： true
- obs[7]: `right_support_contact` — 右支撑腿接触标志，可用于奖励： true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: `no_engine`（不做任何推进/姿态调整）
- action 1: `left_orientation_engine`（启动左姿态引擎，产生逆时针旋转力矩）
- action 2: `main_engine`（启动主引擎，产生与机体朝向一致的推力）
- action 3: `right_orientation_engine`（启动右姿态引擎，产生顺时针旋转力矩）

> 说明：所有引擎动作都会消耗燃料，而燃料总量在观测中不可见。主体通过姿态调整+主引擎实现2D空间的移动。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success‑like termination**：  
  `body_not_awake_or_settled` — 主体静止/稳定，若同时满足 **当前坐标已落在目标垫上**（x,y接近0且接触着陆），视为成功着陆。但环境**未显式标记**成功。
- **failure‑like termination**：  
  `crash_or_body_contact` — 任意身体部分（如支撑腿之外的部位）与地面/障碍物接触，可能表示坠落/翻倒；  
  `horizontal_position_outside_viewport` — 水平位置超出视口边界，表示飞出任务区。
- **ambiguous termination**：  
  仅靠 `body_not_awake_or_settled` 而不检查位置会无法区分“卡在垫外停止”或“悬空静止”的情况，需要结合观测判断。
- **truncation**： 无 episode 长度截断（未在 step 代码中体现）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: 空（info 始终为 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（包括任何 `success`、`failure`、`termination_reason`）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观测，8维数组）
- `action`（当前动作）
- `next_obs`（下一步观测）
- `info` 中**明确允许**的字段（此处无任何允许字段）
- `training_progress` 仅在 prompt 明确要求时使用（本环境未要求，默认忽略）

禁止使用：
- `original_reward` / `official_reward`（已被屏蔽）
- 任何未声明的 `info` 字段
- 任何未在观测空间中说明的切片

## 7. 可用于奖励函数的信号
- **position**：`obs[0]` (x), `obs[1]` (y) 及 `next_obs[0:2]`
- **velocity**：`obs[2]` (vx), `obs[3]` (vy) 及对应 next_obs
- **orientation / angular**：`obs[4]` (角度), `obs[5]` (角速度) 及对应 next_obs
- **contact**：`obs[6]` (左接触), `obs[7]` (右接触) 及对应 next_obs  
  注意：接触标志为 0.0/1.0，可组合判断单脚/双脚接触。
- **action/engine**：`action` 可取 0–3，可映射为燃料消耗（主引擎或姿态引擎使用）
- **other**：无额外环境信息

## 8. 不确定或不可用的信号
- **fuel amount**：未在观测中提供，无法直接获得剩余燃料
- **ground truth success**：无 info 字段，不能依赖外部成功标志
- **absolute pad location**：只有相对位置，但已足够定位着陆垫
- **wind/perturbation**：说明中提到风，但观测中无风信息，且无法用于奖励（除非有速度残差）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body (lander/vehicle)
  actuator_type: main thruster (action 2) + two orientation thrusters (actions 1, 3)
  contact_structure: two support legs (left, right), contact flags in obs[6/7]
primary_objectives:
  - 到达着陆垫中心（x, y 接近 0）
  - 稳定静止在着陆垫上（接近零速度，零角速度，双脚接触）
secondary_objectives:
  - 最小化燃料消耗（减少引擎使用次数）
  - 最小化着陆时间（快速完成）
main_failure_risks:
  - 坠落/翻倒（crash接触）导致任务失败
  - 飞出视口边界
  - 在垫外过早消耗燃料而无法修正
  - 着陆时弹跳或姿态失控而接触不安全
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `approach_progress`  
  purpose: 引导 agent 向目标垫靠近  
  why_required: 导航到达主目标，没有进度信号则可能随机漂移  
  usable_signals: [x_position (obs[0]), y_position (obs[1])]  
  risks: 纯距离奖励可能导致高速撞击，需结合减速约束

- role_id: `soft_landing`  
  purpose: 在接近目标时强制减速并稳定姿态，避免硬着陆  
  why_required: 接触时必须保证低速、小角速度、竖直姿态，以确保安全停靠  
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity]  
  risks: 若始终惩罚速度，可能阻碍早期移动，需设计条件激活（如距离近时才加强）

- role_id: `safe_contact`  
  purpose: 鼓励两脚同时接触且接触发生在目标垫附近  
  why_required: 仅靠位置误差无法区分“悬停在垫上”还是“真正着陆成功”，接触信号提供最终成功证据  
  usable_signals: [left_contact, right_contact, x_position, y_position]  
  risks: 只在已知着陆垫位置时有效；若接触但未在垫上，需惩罚

### 10.2 条件职责 conditional_roles
- role_id: `fuel_efficiency`  
  condition_to_use: 整个 episode 都可使用，作为辅助项  
  usable_signals: [action]  
  risks: 动作惩罚过强可能导致 agent 不做动作、坠落；需要与进度奖励平衡，且不能压制必要的修正动作

- role_id: `time_penalty`  
  condition_to_use: 若希望鼓励快速完成，可在每一步施微小负奖励  
  usable_signals: [constant per step]  
  risks: 典型的步数惩罚，容易使 agent 急于求成而失败，需要弱权重

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `explicit_success_bonus`  
  reason: 依赖 `info["success"]`，但环境中该字段不存在；强行猜测易导致错误信号  
  forbidden_or_missing_signals: [info.success]

- role_id: `remaining_fuel_reward`  
  reason: 观测中不包含燃料量，无法直接获取剩余燃料值  
  forbidden_or_missing_signals: [fuel_remaining]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_progress | x_position, y_position | — | dense_state_signal (e.g. negative L2 distance), bounded_signal (clip large distances) | 作为持续推进信号，可结合速度方向 |
| soft_landing | x_velocity, y_velocity, body_angle, angular_velocity | — | quadratic_penalty (on velocities/angles), bounded_signal (only activate when near target) | 关键：用距离门控，只当接近目标垫时才大幅惩罚大速度/角度 |
| safe_contact | left_contact, right_contact, x_position, y_position | explicit leg-on-pad flag | sparse_event (reward on both legs contact) conditioned on <threshold distance | 必须结合位置检查，避免在垫外接触时误奖 |
| fuel_efficiency | action | fuel_consumption | action_weighted_penalty (penalize engine actions) | 推荐离散动作惩罚：action 1,2,3 均扣罚 |
| time_penalty | step count (implicit) | — | constant_step_penalty | 非常弱的负奖励，避免成为主导 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主引擎常开导致燃料快速耗尽 | 动作2比例极高、终止于 crash 或飞出 | 增加动作惩罚或课程学习先禁用主引擎惩罚，逐步引入 |
| 在目标上方高速旋转无法稳定 | 角速度持续大、角度振荡 | 增加姿态/角速度惩罚，或对角度偏差加入饱和项 |
| 过早减速悬停在目标外 | 速度很早降为零，但距离目标仍远 | 延迟 soft_landing 的角色激活距离，确保先完成 approach |
| 着陆垫外双腿同时接触（虚假成功） | 接触时 x,y 离 0 较远，但已终止 | 在接触奖励中强制要求 x,y 在给定半径内；否则给出惩罚 |
| 飞越目标后返回失败 | x 越过目标后持续正向移动，或 y 下降过快 | 检查进度奖励是否对称，设计非对称引导（越远代价更高） |
| 姿态引擎频繁切换但主引擎不用 | 动作 1,3 交替，动作 2 几乎不发生 | 适当放宽动作惩罚，或为动作序列增加平滑项（可选） |