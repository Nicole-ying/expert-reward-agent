# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 2D 刚体飞行轨迹优化任务。  
**主目标**：从初始位置尽快到达画面中心的固定目标垫，并稳定停靠其上（速度接近于零，姿态保持平稳，左右支撑脚同时接触）。  
**次目标**：在保证成功到达的前提下，尽可能少地使用主引擎和姿态引擎推力。  
**不应混淆**：这**不是**纯粹的生存平衡任务，也不是多目标冲突的权衡——到达停靠是唯一的核心结果，燃料经济性只是“越快/越省”的优化维度，不构成独立的并行目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是“到达指定目标位置并稳定停靠”，存在明确的相对位置（观察空间前两维），终止条件包含“身体未唤醒或已稳定停靠”，动作空间控制推力与姿态，典型的目标导向导航形态。次目标（最少推力）可作为条件优化，但并不是与到达等效的平行目标，因此不选择 multi_objective_task。

动力学子类型：  
dynamics_subtype: goal_approach_and_soft_contact  
理由：飞行器需要趋近目标、减速、保持姿态、并最终以低速、稳定的身体接触停在垫上，完全匹配“接近目标并低速、稳定接触/停靠”的定义。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（源信息未明确，但观测字段多为浮点数，接触标志以浮点 0.0/1.0 表示）
- 各维含义及奖励可用性：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | x_position | 相对于目标垫中心的水平坐标 | true |
| 1 | y_position | 相对于垫面高度的垂直坐标 | true |
| 2 | x_velocity | 水平线速度 | true |
| 3 | y_velocity | 垂直线速度 | true |
| 4 | body_angle | 机体俯仰/倾斜角度 | true |
| 5 | angular_velocity | 机体角速度 | true |
| 6 | left_support_contact | 左支撑脚接触标志（1.0 接触，0.0 未接触） | true |
| 7 | right_support_contact | 右支撑脚接触标志 | true |

注意：所有 obs 均可在奖励函数中使用。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：

| 动作 ID | 名称 | 含义 |
|----------|------|------|
| 0 | no_engine | 不执行任何推力，仅受物理影响 |
| 1 | left_orientation_engine | 点燃一个姿态引擎（产生转矩，改变角度） |
| 2 | main_engine | 点燃主引擎（产生沿当前朝向的推力） |
| 3 | right_orientation_engine | 点燃相反姿态引擎（与动作 1 反向的转矩） |

动作在奖励函数中可用，例如用于计算推力消耗的惩罚。

## 5. step 与终止条件分析
### 5.1 终止模式
源 step 中定义：
```python
terminated = crash_or_body_contact or horizontal_position_outside_viewport or body_not_awake_or_settled
```

- **crash_or_body_contact**：发生碰撞或机身其他部位（非支撑脚）接触地面 → 可能为**失败**。
- **horizontal_position_outside_viewport**：水平位置偏离视口 → **明确失败**。
- **body_not_awake_or_settled**：身体进入休眠状态（失败）或已稳定停靠（成功）。该条件同时包含成功与失败两种情况，需根据状态推断 → **模棱两可的终止**。

此外未设置截断（truncation 恒为 False）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available: false**
  - info 字典为空，没有类似 `"success"` 的布尔标志。
- **explicit_failure_flag_available: false**
  - 虽然 crash 和出界显然为失败，但 step 源码未提供单独标志。
- **allowed_info_fields**: 空（info={}），奖励函数中**禁止**依赖任何 info 键。
- **forbidden_or_uncertain_info_fields**:
  - 任何未出现在源码中的 info 字段均禁止使用（如 `success`, `failure_reason` 等）。
  - 终止条件 `body_not_awake_or_settled` 未作为信号传入奖励，故无法直接区分，必须通过下一条终止时的 next_obs 特征（位置、速度、接触等）自行推测成功与否。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：上一步观察（所有维度）
- `action`：被执行的离散动作（0~3）
- `next_obs`：步骤后的观察（所有维度）
- 无其他可用 info 字段（info 为空）

禁止使用：
- `original_reward`：官方奖励被屏蔽，严禁依赖
- `info` 中任何未声明的键
- `training_progress`：当前提示未明确允许使用，故禁止（若后续明确可用则允许）

在不违反上述规则的前提下，可以利用 `next_obs` 在最后一步计算终止奖励（如对成功停靠给予大正奖励、对碰撞或出界给予负奖励）。

## 7. 可用于奖励函数的信号
以下归纳主要信号组：

- **位置相关**：`x_position`（水平偏差）、`y_position`（垂直高度）
- **速度相关**：`x_velocity`、`y_velocity`
- **姿态相关**：`body_angle`、`angular_velocity`
- **接触相关**：`left_support_contact`、`right_support_contact`
- **动作/引擎使用**：`action`（可解析为主引擎、姿态引擎是否点火）
- **其他**：可构造的组合信号，如到目标的欧氏距离、速度范数、角度误差、是否双腿同时接触（ `left_support_contact > 0.5 and right_support_contact > 0.5` ）作为停靠标志。

## 8. 不确定或不可用的信号
- **成功/失败标志**：不存在显式 flag，需根据终止条件推定
- **目标垫半径或停靠判定阈值**：环境描述未给出精确的几何尺寸与速度阈值，需通过观察大量轨迹或保守假设（如距离 < 0.1，速度 < 0.05，角度 < 0.1，双腿接触）来推断成功
- **燃料量或推力大小**：环境中推力大小被屏蔽，只能以二元动作是否发生作为粗略指标（消耗主引擎/姿态引擎的次数）
- **阶段信息（进度）**：无 progress 字段，training_progress 当前禁止使用
- **环境原始奖励**：已屏蔽，不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 刚体飞行器
  actuator_type: 一个主推力引擎、两个差动姿态引擎
  contact_structure: 左右两个支撑脚，接触垫面后支撑机身，要求平稳双足触地
primary_objectives:
  - 水平与垂直位置收敛至目标垫区域（距离极小）
  - 速度衰减至近乎为零（软着陆）
  - 姿态角稳定至竖直附近（防止倾覆）
secondary_objectives:
  - 最小化主引擎与姿态引擎的使用次数（降低推力消耗）
  - 尽快完成停靠（隐含在速度与距离衰减中）
main_failure_risks:
  - 过早与地面碰撞或机身其他部分触地（crash）
  - 水平漂移出界
  - 速度过大或单侧脚先着地导致翻倒/未稳定
  - 姿态角发散，无法控制翻转
  - 为了省力而过度悬停，长时间无法停靠
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: approach_target**  
  purpose: 驱使飞行器向目标垫靠近，缩小水平与垂直偏差。  
  why_required: 到达目标垫是任务的核心结果。  
  usable_signals: `x_position`, `y_position`（或组合成距离）  
  risks: 惩罚过重可能导致急加速撞地；需与速度控制协调。

- **role_id: soft_landing**  
  purpose: 在靠近目标时强制减速，避免高速冲击导致 crash 或不稳定接触。  
  why_required: 停靠不仅要求位置正确，还要求速度极低。  
  usable_signals: `x_velocity`, `y_velocity`（速度范数）  
  risks: 过早强制减速可能延长训练时间；需结合距离权重，仅在接近目标时增强。

- **role_id: stable_upright**  
  purpose: 保持机体姿态接近竖直，防止因倾斜导致单脚着地或翻倒。  
  why_required: 稳定停靠需要双足同时接触，姿态偏差过大会破坏这一条件。  
  usable_signals: `body_angle`, `angular_velocity`（以及双脚接触标志作为验证）  
  risks: 过度强调可能使动作选择保守；可仅当接近垫面时提升权重。

### 10.2 条件职责 conditional_roles
- **role_id: fuel_efficiency**  
  purpose: 最小化推力使用，避免不必要的引擎点火。  
  condition_to_use: 当飞行器已经学会基本到达和减速后，或者通过训练进度动态调控；初期训练可关闭或降低权重，防止学习停滞。  
  usable_signals: `action`（解析为主引擎、姿态引擎是否点火）  
  risks: 过早引入会与到达、减速目标冲突，导致 agent 选择 no_engine 而悬停不前。

- **role_id: terminal_success_bonus**  
  purpose: 在 episode 结束时，若能识别成功停靠，给予一次性大幅奖励。  
  condition_to_use: 必须依据 `next_obs` 推断是成功停靠而非 crash/出界；简单判定条件：`abs(x_position) < 阈值, y_position < 阈值, sqrt(x_vel^2+y_vel^2) < 阈值, abs(angle) < 阈值, 左右接触 > 0.5`。  
  usable_signals: `next_obs` 的上述组合  
  risks: 阈值必须保守，否则可能误判 crash 为成功；应同时配合失败惩罚。

- **role_id: terminal_failure_penalty**  
  purpose: 对所有明显失败（水平出界、速度过大撞击）给予负奖励，防止 agent 落入不安全区域。  
  condition_to_use: 从 `next_obs` 能够确认的危险状态（如 `abs(x_position)` 非常大、`y_position` 突变、速度极大且无接触等）。由于无显式 crash 标志，需谨慎设计启发式条件。  
  usable_signals: `next_obs` 的位置、速度、接触  
  risks: 误将成功案例惩罚，破坏学习。

### 10.3 慎用/禁用职责 avoid_roles
- **role_id: early_contact_penalty**  
  reason: 本任务中双腿同时接触是停靠成功的一部分，对过早的单腿接触直接惩罚可能阻止 agent 学习降落；应在接触时检查是否满足成功条件。  
  forbidden_or_missing_signals: 缺少真正的“非稳定接触”标志，无法区分 landing 过程中的短时单脚触地和碰撞。

- **role_id: angular_smoothness**  
  reason: 现阶段任务关注终点姿态误差，而非过程角速度平滑。角速度惩罚与姿态稳定职责部分重叠，过度施加可能限制必要的姿态调整。  
  forbidden_or_missing_signals: 无直接的不良振荡度量。

## 11. role_to_signal_mapping

| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | `x_position`, `y_position` (obs/next_obs) | 无 | dense_state_signal (负距离), bounded_signal | 可用 `-sqrt(x^2+y^2)` 或分段线性 |
| soft_landing | `x_velocity`, `y_velocity` (obs/next_obs) | 无 | dense_state_signal (负速度范数), quadratic_penalty | 可结合与目标的距离作为乘数 |
| stable_upright | `body_angle`, `angular_velocity` | 无 | bounded_signal (负绝对值角度), damping_penalty | 角度误差小于阈值时不处罚 |
| fuel_efficiency | `action` (离散) | 推力大小 | per_action_binary_penalty | 对 action 1,2,3 施加小负奖励 |
| terminal_success_bonus | `next_obs` (位置、速度、角度、接触) | explicit_success_flag | large_discrete_bonus_upon_condition | 判断条件见 10.2 |
| terminal_failure