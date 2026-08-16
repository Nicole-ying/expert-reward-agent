# Response Record

# 匿名环境理解卡片

## 1. 任务目标
任务主体是一个 2D 刚体飞行器/着陆器，从视口上方中央附近出发，初始受随机力。
**主目标**：尽快到达并稳定停靠在场景中央的目标垫上，实现双腿同时接触、姿态平稳、速度极低，且全程安全不碰撞、不越界。
**次目标**：尽量少用引擎推力（燃料节省），隐式要求动作决策高效。
**不应混淆的目标**：单纯追求极致的速度而不计安全，或只追求零耗能而停滞不前。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching
- confidence: high  
- reason: 核心是到达并停稳在特定目标位置（目标垫），附属的速度、能耗、姿态要求都是围绕这一主要目标展开，不存在多个权重相当、彼此互斥且无法区分主次的目标，因此属于导航目标到达族。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position，相对于目标垫的水平坐标，reward_usable: true
- obs[1]: y_position，相对于目标垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，机体朝向角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（0/1），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志（0/1），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不点火，无推力
- action 1: left_orientation_engine — 启动左侧姿态调节引擎
- action 2: main_engine — 启动主引擎（提供主要推力）
- action 3: right_orientation_engine — 启动右侧姿态调节引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  body_not_awake_or_settled 可能表示机体稳定静止（速度、角速度极小），配合位置接近目标和双支撑腿接触，大概率是成功着陆。  
  但该条件本身不检查位置，可能提前在错误位置稳定，因此属于模糊终止。
- failure-like termination:  
  crash_or_body_contact（机体或部件异常碰撞，如头部触地），  
  horizontal_position_outside_viewport（水平越界）。
- ambiguous termination:  
  body_not_awake_or_settled（如上述）
- truncation: 无

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: {} （空字典，无可用字段）  
- forbidden_or_uncertain_info_fields:  
  - info 整体不可用，不应依赖任何 info 键值  
  - 无法直接知晓官方奖励值（被遮蔽）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs, action, next_obs（全部合法位置通道）
- info 仅限空字典，无可用字段
- training_progress 仅在明确需要时可用，本环境未强制要求使用

禁止使用：
- original_reward（已遮蔽）
- 任何未声明的 info 字段
- 任何对 obs 的越界或未说明切片

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position) — 直接度量到目标的距离
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) — 可控制着陆冲击与稳定性
- orientation: obs[4] (body_angle) — 姿态控制，obs[5] (angular_velocity) — 旋转稳定性
- contact: obs[6] (left_support_contact), obs[7] (right_support_contact) — 双腿着地指示
- action/engine: action (离散 0~3) — 可构建燃料惩罚或推力惩罚
- other: 无

## 8. 不确定或不可用的信号
- 官方成功/失败标志：不存在  
- 环境给出的稀疏终局奖励：被遮蔽  
- 时间步数/剩余时间：不可见  
- 目标垫的确切接触力/冲击力：不可见  
- 风场等外部扰动：未知  
- 任何 info 内容：不可用

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: rigid_body (2D vehicle/lander)
  actuator_type: main_thruster, left_orientation_thruster, right_orientation_thruster
  contact_structure: two_support_legs (left/right) with binary contact sensors
primary_objectives:
  - reach_target_pad (x ≈ 0, y ≈ 0)
  - establish_stable_biped_contact (both legs on ground, low velocity, small angle)
secondary_objectives:
  - minimize_engine_usage (fuel efficiency)
  - minimize_time_to_settle (fast settling)
main_failure_risks:
  - crashing or touching ground with non-leg parts
  - drifting outside horizontal boundaries
  - settling prematurely away from the pad
  - flipping or large orientation deviation
  - excessive fuel consumption preventing arrival
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 驱使智能体向目标垫中心移动。  
  why_required: 主目标为到达，无此奖励无法形成到达行为。  
  usable_signals: [obs[0], obs[1]]  
  risks: 可能导致高速直接撞击而非软接触，需配合速度控制。

- role_id: velocity_control  
  purpose: 靠近目标时减小速度，避免剧烈碰撞或反弹。  
  why_required: 着陆安全与稳定需要低速，否则可能跳起或损坏。  
  usable_signals: [obs[2], obs[3], obs[0], obs[1]（用于距离门控）]  
  risks: 过早减速可能延长到达时间。

- role_id: stable_contact  
  purpose: 鼓励双腿同时触地并维持接触，防止单腿着陆或翻倒。  
  why_required: 任务要求“make safe contact”且稳定停靠，双支撑是安全着陆的关键特征。  
  usable_signals: [obs[6], obs[7]]  
  risks: 可能诱导故意摔下以获得接触，需要结合位置与速度约束。

- role_id: attitude_stabilization  
  purpose: 保持机体朝向接近竖直，防止过大倾角导致侧翻。  
  why_required: 着陆稳定和引擎安全要求姿态可控。  
  usable_signals: [obs[4], obs[5]]  
  risks: 过度严厉的姿态奖励可能妨碍姿态调节机动。

- role_id: fuel_efficiency  
  purpose: 惩罚不必要的引擎使用，节省能耗。  
  why_required: 次目标明确要求“as little engine thrust as possible”，是显著优化方向。  
  usable_signals: [action]  
  risks: 权重过高会阻碍跨越距离，或导致长时间滑翔不点火。需与 proximity 奖励平衡。

### 10.2 条件职责 conditional_roles
- role_id: fast_settling_incentive  
  condition_to_use: 当智能体已能较稳定到达目标垫后加入，或训练中后期使用，加速收敛时间。  
  usable_signals: 随时间步隐含（通过奖励标量衰减），或结合距离门控的小幅每步惩罚。  
  risks: 过早加入会使初期探索因持续惩罚而失败；需要配合目标到达奖励，避免鼓励草率着陆。

### 10.3 慎用/禁用职责 avoid_roles
- 无特别禁用的角色，但应避免使用任何依赖 info 或官方奖励的角色，因为相关信号不可用。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | obs[0], obs[1] | 无 | distance_penalty, potential-based shaping | 通常用负欧几里得距离或其平方 |
| velocity_control | obs[2], obs[3] (门控用 obs[0], obs[1]) | 无 | squared penalty, bounded_signal | 靠近目标时加重对速度的惩罚 |
| stable_contact | obs[6], obs[7] | 无 | product (both_contact=left*right), binary bonus | 可配合速度接近零时给予正奖励 |
| attitude_stabilization | obs[4], obs[5] | 无 | angle_penalty, quadratic_penalty | 以角度的平方或绝对值惩罚 |
| fuel_efficiency | action (0~3) | 无 | action_cost_vector (0 cost for no_engine, positive for others) | 主引擎和姿态引擎可设置不同代价 |
| fast_settling_incentive | 隐含 (step counter) | 显式时间信号 | small_constant_penalty | 用极小负数每步施加，需谨慎调权 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 无法接近目标，远距离徘徊 | 平均 episode 末端 |x|,|y| 仍然很大，轨迹发散 | 加强 goal_proximity 权重，加入电位塑形 |
| 高速撞击目标垫 | 着陆瞬间 |vy| 大，接触后跳起或翻倒 | 提升 velocity_control 权重，并在接近时启用强力减速惩罚 |
| 单腿着地后翻倒 | left*right 很少同时为1，结束前 body_angle 剧烈变化 | 增加双接触奖励，并严格惩罚大 angular velocity |
| 无明显理由的越界或碰撞 | 轨迹在视口边缘频繁 crash | 检查 initial force 随机扰动过大，或 guidance 奖励不足导致偏航；可引入轻微居中引导 |
| 过量使用引擎，燃料耗尽仍无法到达 | 动作统计中 0 号动作使用极少，总步数短 | 提升 fuel_efficiency 惩罚，同时确保 goal_proximity 能引导高效路径 |
| 过早沉降（稳定在远离目标处） | episode 因 settled 终止，但 |x| 或 |y| 很大 | 将 settled 终止视为无效，需配合位置条件；强化到达奖励 |
