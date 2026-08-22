# 匿名环境理解卡片

## 1. 任务目标
本环境是一个二维平面内的轨迹优化问题。智能体控制的类飞行器从视口顶部中心附近以随机初速度出发，核心目标是**快速且平稳地到达并停靠在中央目标垫上**。  
为了实现这一主目标，智能体需要学会：  
- 向中央目标区域靠近  
- 逐步降低运动速度  
- 维持稳定的朝向  
- 在目标垫上实现安全、低速的接触与稳定  

**次要目标**（应服从主目标）是在满足成功到达的前提下**尽可能少消耗引擎推力**。**不应混淆的目标**包括单纯追求极低能耗而放弃速度控制，或为了快速到达而引发猛烈坠地。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心是到达指定目标位置并稳定停靠，所有行为（接近、减速、姿态保持）都服务于该最终空间目标。燃料消耗最小化是明确的次要优化方向，但权重上属于辅助指标，不属于多目标冲突场景。

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32 (假设，原始字段未声明，但连续值通常如此)  
- 各维度含义（index 从 0 开始）：
  - obs[0]: x_position，相对于目标垫的水平坐标，usable for reward: true  
  - obs[1]: y_position，相对于目标垫高度的垂直坐标，usable for reward: true  
  - obs[2]: x_velocity，水平线速度，usable for reward: true  
  - obs[3]: y_velocity，垂直线速度，usable for reward: true  
  - obs[4]: body_angle，机体朝向角，usable for reward: true  
  - obs[5]: angular_velocity，角速度，usable for reward: true  
  - obs[6]: left_support_contact，左侧支撑接触标志（0/1），usable for reward: true  
  - obs[7]: right_support_contact，右侧支撑接触标志（0/1），usable for reward: true  

注：所有字段均为环境直接提供，reward 函数中可以全部使用。

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义：
  - action 0: `no_engine` – 不激活任何引擎，依靠惯性滑行  
  - action 1: `left_orientation_engine` – 点燃左侧姿态引擎，产生转向力矩（推测可使机体逆时针旋转）  
  - action 2: `main_engine` – 点燃主引擎，产生指向机体正向的推力（用于减速或抬升）  
  - action 3: `right_orientation_engine` – 点燃右侧姿态引擎，产生反向转向力矩（推测使机体顺时针旋转）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` 在**同时满足**位置接近目标、速度极低、双足接触目标垫的情况下，很可能代表成功着陆并稳定。  
- failure-like termination:  
  - `crash_or_body_contact`：机体或任何部位与地面/障碍发生非预期接触，视为坠毁或硬着陆。  
  - `horizontal_position_outside_viewport`：机体横向飞出允许范围，视为失控。  
- ambiguous termination:  
  `body_not_awake_or_settled` 出现在未到达目标区域或接触状态异常时，可能是中途卡死或坠落失败；需要结合下一状态观察区分成功/失败。  
- truncation:  
  源 step 代码中返回 `truncated=False`，无截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空字典，无 success 字段）  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info={}）  
- forbidden_or_uncertain_info_fields: 任何未在 step 源码中出现的字段均禁止使用（如 `success`, `done_reason`, `reward_components` 等）

因此奖励函数不能依赖 info 来获知成功或失败，必须基于 next_obs 的信号自行判断。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- `obs`：完整的 8 维观测向量
- `action`：刚才执行的动作（0-3）
- `next_obs`：执行后的 8 维观测向量
- `info`：允许使用，但本环境始终为 `{}`
- `training_progress`：仅在 prompt 明确说明可用时才使用；本任务未提及，应**禁用**或保持为占位参数

禁止使用：
- `original_reward`：已被掩码，不可使用
- 任何未声明的 info 字段
- 未声明的 obs 切片（目前所有 8 维均已声明，故均可使用）

## 7. 可用于奖励函数的信号
- position: next_obs[0] (相对目标垫的水平距离)、next_obs[1] (相对垫高度)  
- velocity: next_obs[2] (水平速度)、next_obs[3] (垂直速度)  
- orientation: next_obs[4] (朝向角)、next_obs[5] (角速度)  
- contact: next_obs[6] (左腿接触)、next_obs[7] (右腿接触)  
- action/engine: `action` 本身可用于判断是否开启主引擎或姿态引擎  
- other: 可从上述信号推导出“已安全着陆”的复合条件（如位置接近零、速度接近零、双足均接地）

## 8. 不确定或不可用的信号
- 绝对成功/失败标志：不存在  
- 中间奖励或官方奖励：被掩码，不可用  
- 能量消耗/推力大小：动作空间是离散的，缺少连续推力值，仅能通过动作类型估计引擎使用情况  
- 燃料剩余：无相关观测  
- 目标垫宽度/形状：未明确给出，需从接触条件推断  
- 风或扰动：步骤中提及 wind 但被省略，无法可靠使用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: planar_rigid_body_with_two_contact_points
  actuator_type: main_thruster_plus_two_orientation_thrusters
  contact_structure: two_ground_contacts_left_and_right
primary_objectives:
  - reach_and_settle_at_target_pad
  - maintain_stable_orientation_during_descent
secondary_objectives:
  - minimize_engine_usage (fuel conservation)
  - achieve_task_quickly (implicit via episode termination upon success)
main_failure_risks:
  - crashing or making hard ground contact away from pad
  - drifting horizontally out of viewport
  - oscillating and never stabilizing
  - failing to reduce velocity before contact
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: `approach_target`
  purpose: 引导智能体向目标垫的水平方向和垂直高度靠近
  why_required: 达成目标位置是最核心的主任务，不可缺失
  usable_signals: [next_obs[0], next_obs[1]]
  risks: 单纯追求近距离可能导致高速冲向目标，因此必须与减速职责配合

- role_id: `velocity_damping`
  purpose: 在靠近目标过程中以及最终着陆时，惩罚过高的线速度，鼓励平稳减速
  why_required: 没有减速将导致硬着陆或飞越目标，确保安全接触
  usable_signals: [next_obs[2], next_obs[3]]
  risks: 过早强调减速可能使智能体不敢移动；需要与距离条件组合使用

- role_id: `orientation_stabilization`
  purpose: 保持机身朝向接近竖直（或指定安全姿态），避免旋转和侧翻
  why_required: 姿态失控会增加坠毁风险，且接触垫时需要双足同时接地
  usable_signals: [next_obs[4], next_obs[5]]
  risks: 过度惩罚角速度可能阻碍必要转向，需在接近目标时加强

- role_id: `soft_landing`
  purpose: 最终着陆瞬间给予奖励，使智能体以最低速度双足同时接触垫子并稳定
  why_required: 这是“安全接触”的具体化，标志任务成功结束
  usable_signals: [next_obs[6], next_obs[7]; 复合条件：|next_obs[:2]|<threshold, |next_obs[2:4]|<threshold, both_legs_contact==1.0]
  risks: 作为稀疏奖励可能难以学习，需要与上述密集信号协同

### 10.2 条件职责 conditional_roles
- role_id: `fuel_efficiency_penalty`
  condition_to_use: 在智能体已经稳定接近目标区域（例如距离小于某阈值）时启用；大距离时不应启用，以免阻碍快速移动
  usable_signals: [action]
  risks: 可能导致智能体完全不使用引擎，无法减速或调整姿态；必须与主职责平衡

### 10.3 慎用/禁用职责 avoid_roles
- role_id: `success_exclusive_bonus` （基于显式成功标志的固定大奖励）
  reason: 环境不提供显式 success 信号，无法可靠实现；基于自建成功的判定可能错误地将失败判为成功
  forbidden_or_missing_signals: [explicit_success_flag]

- role_id: `time_step_penalty` （每步固定扣分以鼓励快速完成）
  reason: 环境使用 `terminated` 立即结束，快速完成已经通过提前结束自然体现；强制每步惩罚可能与省燃料职责冲突，且因其恒定性难以区分成功前的必须步数和失败时的浪费步数
  forbidden_or_missing_signals: 无合适信号量化“浪费时间”，且与主任务存在倾向冲突，慎用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_target | next_obs[0], next_obs[1] | None | dense_state_signal (distance to origin), bounded_signal (goal region) | 可用欧氏距离或曼哈顿距离 |
| velocity_damping | next_obs[2], next_obs[3] | None | quadratic_penalty, scaled_penalty with distance gating | 惩罚需随距离减小而增强 |
| orientation_stabilization | next_obs[4], next_obs[5] | None | quadratic_penalty on angle error, angular velocity penalty | 角度误差应定义为与竖直方向的差异 |
| soft_landing | next_obs[6], next_obs[7] and filtered position/velocity condition | explicit success flag | sparse_event_reward (condition-based) | 只能由 next_obs 组合判定，需设置合理阈值 |
| fuel_efficiency_penalty | action (only main/ side engine fired) | continuous thrust magnitude | action_mask_cost (e.g., penalty if action!=0) | 仅在目标附近启用，避免阻碍初始移动 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体仅悬停不动，不向目标移动 | episode 步数很多但位置几乎不变，最终因时间过长或未接触终止 | 降低速度惩罚权重，增加距离引导的引力项 |
| 高速冲向目标垫并坠毁 | 垂直速度很大，或终止于 crash 标志，接触时双腿未同时接地 | 加强 velocity_damping 和 soft_landing 条件奖励，增大速度惩罚系数 |
| 持续摆荡，无法平稳 | 角速度长期非零，x 位置来回震荡 | 增强姿态惩罚，加入累积角度偏差项或动作平滑约束 |
| 过早点燃主引擎导致上升远离目标 | y 坐标持续增大，远离零 | 添加对远离目标高度增加的负奖励（仅当向上时） |
| 仅使用姿态引擎旋转而不前进 | action 多为1或3，主引擎未用，位置不动 | 可能速度惩罚过强，可允许在远离目标时减少对主引擎使用的惩罚，或给予前进动力激励 |