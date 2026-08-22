# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境为一个二维动力学轨迹优化任务。一个带支撑腿的飞行器从视口顶部中央附近以随机初始力出发，目标是以最快速度、最低引擎消耗到达画面中央的目标平台并稳定停靠。智能体必须学会接近目标、消减速速度、保持直立姿态，并使双腿安全接触平台。 **主要目标是成功着陆并保持稳定，次要目标是缩短用时与节省燃料**。注意：绝不能将燃料节省或速度优化视作比安全着陆更重要的指标。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**
confidence: high
reason: 核心目标是到达指定目标位置（中央平台）并稳定停止，属于典型的点到点导航类目标，而非单纯持续前进或永续平衡。附属的燃料/时间优化是典型的附加约束，不构成平行多目标。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 假设 float32（任务未指定，但连续元素通常如此）
- 各维度含义（可直接用于奖励计算）：
  - obs[0]: `x_position`，相对于目标平台的水平坐标，reward_usable: true
  - obs[1]: `y_position`，相对于平台高度的垂直坐标，reward_usable: true
  - obs[2]: `x_velocity`，水平线速度，reward_usable: true
  - obs[3]: `y_velocity`，垂直线速度，reward_usable: true
  - obs[4]: `body_angle`，机身倾斜角，reward_usable: true
  - obs[5]: `angular_velocity`，角速度，reward_usable: true
  - obs[6]: `left_support_contact`，左支撑腿接触标志（1.0/0.0），reward_usable: true
  - obs[7]: `right_support_contact`，右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - 0: `no_engine` – 不启动任何引擎
  - 1: `left_orientation_engine` – 启动左侧姿态引擎
  - 2: `main_engine` – 启动主引擎（提供推力，通常方向与机体方向有关）
  - 3: `right_orientation_engine` – 启动右侧姿态引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **body_not_awake_or_settled** 可能与成功稳定着陆高度相关（身体静止在平台上），但需结合双腿接触和位置判断，并非绝对成功信号。
- failure-like termination:
  - **crash_or_body_contact**：机身与地面或其它危险区域碰撞，极可能代表坠毁。
  - **horizontal_position_outside_viewport**：机体水平坐标超出允许范围，必然失败。
- ambiguous termination: **body_not_awake_or_settled** 既可能代表成功稳定着陆，也可能代表卡死在非落地状态，需通过位置/接触进一步区分。
- truncation: 无明确截断时间限制（任务描述未提供），但环境可能暗含步数限制；当前不得假设截断信号可用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**（info 为空，没有 success 字段）
- explicit_failure_flag_available: **false**
- allowed_info_fields: 无（info = {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在，禁止使用 `info["success"]`、`info["failure"]` 等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
允许使用：
- `obs` 全部 8 维
- `action` 整数值
- `next_obs` 全部 8 维（用于判断终止状态或变化）
- `training_progress` 仅在 prompt 明确允许时方可使用（本任务未提及，建议**禁止**使用）

禁止使用：
- `original_reward` — 官方奖励已被屏蔽，绝对不可读取
- `info` 中任何字段 — info 为空字典，没有任何可用字段
- 未在 observation_space 中声明的任何切片

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`，可以计算与目标(0,0)的距离或直接分量。
- **velocity**: `x_velocity`, `y_velocity`，合成速度大小，或垂直速度惩罚。
- **orientation**: `body_angle`, `angular_velocity`，用于保持直立。
- **contact**: `left_support_contact`, `right_support_contact`，作为着陆完成的判据之一。
- **action/engine**: 可通过 action 值判断是否使用引擎，用于燃料惩罚。
- **other**: 可从上述信号组合出距目标距离、是否接近、是否静止、是否双腿着地且低速等衍生信号。

## 8. 不确定或不可用的信号
- 终止原因（如 `info["termination_reason"]`）不可用。
- 目标平台的具体坐标或范围未给出，但从观察中 `x_position` 和 `y_position` 的定义可知目标平台位于 (0, 某高度?)，观察值已经是相对量，因此无需额外提取。
- 任何与 wind、外部力相关的信息已被屏蔽。
- 训练步数/进度不明确，不应依赖 `training_progress`。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid body with two symmetrical landing legs
  actuator_type: main engine + two orientation thrusters
  contact_structure: two support legs (left, right) providing binary contact flags
primary_objectives:
  - Reach the target pad (minimize horizontal and vertical position error)
  - Achieve stable, low-velocity landing with both legs on ground
secondary_objectives:
  - Minimize fuel consumption (reduce engine usage)
  - Complete the task quickly (implicit through smooth approach)
main_failure_risks:
  - Crashing the body on the ground causing destruction
  - Drifting horizontally out of the viewport
  - Approaching too fast and bouncing off the pad
  - Overusing orientation engines and losing stability
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: approach_target**
  - purpose: 引导飞行器减小到目标平台的距离。
  - why_required: 任何有效策略都必须向目标移动，否则无法完成任务。
  - usable_signals: [`x_position`, `y_position`] 或它们的组合（如欧氏距离）
  - risks: 若单独使用可能导致高速俯冲并撞毁，必须和其他职责配合。

- **role_id: soft_landing**
  - purpose: 确保着陆瞬间及最终阶段速度低、姿态稳定且双腿触地。
  - why_required: "settle at a central target pad" 明确要求稳定接触，仅到达位置不够。
  - usable_signals: [`x_velocity`, `y_velocity`, `body_angle`, `angular_velocity`, `left_support_contact`, `right_support_contact`] 以及 next_obs 中的相同维度。
  - risks: 过度强调低速可能导致 never-transition-to-contact 或过早悬停。

- **role_id: success_reward**
  - purpose: 在成功着陆终止时给出明确正反馈，强化完整序列。
  - why_required: 终止条件中无显式成功信号，必须由奖励函数根据 `next_obs` 在终止时判断成功状态（双腿着地、低速、位置偏差小）。
  - usable_signals: 使用 `next_obs`（或结合 `obs`）的接触标志、速度、位置。通常仅在 `terminated=True` 且达到成功条件时施加。
  - risks: 定义苛刻容易导致稀疏，定义宽松可能鼓励奖励黑客。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_penalty**
  - condition_to_use: 始终可用，通常作为每步的轻微惩罚。
  - usable_signals: `action`（当 action != 0 时施加负奖励）
  - risks: 惩罚过大会导致智能体完全不使用引擎，无法着陆；过小则无效。可与进度奖励一起使用。

- **role_id: progress_speed_encouragement**
  - condition_to_use: 如果希望压缩时间，可引入基于向目标靠近速度的正 shaping 或每步小负惩罚来加速收敛。但当环境有截断步数限制时更安全；本描述未提及步限，谨慎使用或设为弱权重。
  - usable_signals: 距离的减少量 (`prev_dist - curr_dist`)，或速度朝向目标的分量。
  - risks: 若步数无上限，每步负奖励可能导致过早结束或焦虑策略。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: official_reward_clone**
  - reason: 官方奖励被屏蔽，严禁使用 `original_reward`。
  - forbidden_or_missing_signals: `original_reward`

- **role_id: info_success_based_reward**
  - reason: info 为空，没有 success flag，无法直接使用终止原因。
  - forbidden_or_missing_signals: `info["success"]`, `info["failure"]`

- **role_id: pure_exploration_bonus**
  - reason: 环境观测连续且提供丰富 dense 信号，不需要依赖好奇心或访问计数。过早引入可能分散目标。
  - forbidden_or_missing_signals: 探索启发式所需的状态计数等不含在标准输入中。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | x_position, y_position | – | `quadratic_penalty`, `shaped_distance` | 可为(距离的相反数)或-sqrt(x^2+y^2) |
| soft_landing | x_vel, y_vel, body_angle, ang_vel, contact flags | – | `velocity_penalty`, `angle_penalty`, `contact_bonus` | 可在接近目标时放大权重 |
| success_reward | next_obs 的 contact, vel, position | explicit success flag | `large_positive_on_condition` (双腿着地 & 低速 & 位置误差小) | 只在 terminated=True 且满足条件时施加 |
| fuel_penalty | action | – | `constant_penalty_per_engine_use` | action != 0 时恒负，很小 |
| progress_speed_encouragement | 上一帧和当前帧的距离差 | – | `potential_based_shaping` | 使用 `gamma*dist_next - dist` |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不降落 | y_position 保持较大正值，速度接近0，长时间未使用 main engine 或微弱使用 | 增加 fuel_penalty 权重，或引入高度 negative shaping 促使下降 |
| 高速撞击平台 | 终止前一瞬间 y_velocity 极大负值，crash 终止，双腿未接触或接触即毁 | 大幅加强 soft_landing 中垂向速度惩罚，在低高度时权重指数上升 |
| 翻滚失稳后坠毁 | body_angle 与 angular_velocity 剧烈震荡，crash 终止 | 提高 body_angle 和 ang_vel 的惩罚，尤其在接近地面时 |
| 超出水平视口 | x_position 不断增大，最终被 horizonal_outside_viewport 终止 | 加强 approach_target 中水平位置惩罚，特别限制 x_position 绝对值 |
| 过度使用方向引擎摇摆 | 动作1、3频繁切换，x_velocity 左右震荡，燃料消耗巨大 | 微调 fuel_penalty，加入左右动作的平滑性约束（如连续动作差异惩罚） |
| 几乎成功但忽略左/右腿接触 | 一只腿未触地即终止，settled 被误判为成功 | 细化 success 条件，必须双腿接触，调整接触标志权重 |
