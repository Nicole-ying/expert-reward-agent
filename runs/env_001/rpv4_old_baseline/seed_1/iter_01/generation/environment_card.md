# 匿名环境理解卡片

## 1. 任务目标
主体为一个2D飞行器（vehicle-like），初始位置在视口顶部中央附近，带有随机初始作用力。  
核心目标是 **尽快到达并稳定停靠在中央目标着陆平台上**，同时尽可能少使用引擎推力。  
智能体需要学习：趋近目标、降低线速度、保持姿态竖直、实现两腿同时安全接触（软着陆）。  
不应混淆的目标：单纯求快但忽略安全着陆，或一味省燃料而无法到达目标。  

## 2. 任务类型选择
**selected_route_id**: `navigation_goal_reaching`  
**confidence**: high  
**reason**: 任务核心是到达指定目标位置（中央着陆平台）并完成稳定停靠，符合“导航目标到达”族定义；附属的省燃料、动作小等属于次要优化目标，不构成多目标冲突的主体。  

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: [8]  
- **dtype**: 默认为 float64 或 float32（取决于环境实现，但通常为 float64）  
- **obs[0]**: `x_position`，水平方向相对于目标着陆平台中心的偏移量，可用于奖励趋近目标，reward_usable: true  
- **obs[1]**: `y_position`，垂直方向相对于平台高度（接触面）的偏移量，reward_usable: true  
- **obs[2]**: `x_velocity`，水平线速度，reward_usable: true  
- **obs[3]**: `y_velocity`，垂直线速度，reward_usable: true  
- **obs[4]**: `body_angle`，机体倾角（如弧度），reward_usable: true  
- **obs[5]**: `angular_velocity`，角速度，reward_usable: true  
- **obs[6]**: `left_support_contact`，左支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  
- **obs[7]**: `right_support_contact`，右支撑脚接触标志（1.0 接触，0.0 未接触），reward_usable: true  

## 4. 动作空间 action_space
- **type**: Discrete  
- **n**: 4  
- **动作/索引 0**: `no_engine` (不做任何事)，语义：无推力，用于滑行或停靠后保持  
- **动作/索引 1**: `left_orientation_engine` (左姿态引擎)，语义：产生逆时针或顺时针旋转力矩（具体方向取决于环境）  
- **动作/索引 2**: `main_engine` (主引擎)，语义：产生纵向（向上）推力  
- **动作/索引 3**: `right_orientation_engine` (右姿态引擎)，语义：产生与左引擎相反的旋转力矩  

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` 可能表示机体静止稳定，通常意味着已着陆并静止，可能视为成功；  
- **failure-like termination**: `crash_or_body_contact` （如机身碰撞地面或平台以外部分）、`horizontal_position_outside_viewport` （漂出水平边界）很可能表示失败；  
- **ambiguous termination**: `crash_or_body_contact` 若接触平台但判定为 crash 则为失败，但描述未区分成功接触与失败接触；我们需要从“到达并稳定停靠”推断理想行为是两腿接触且低速，但不能直接从终止信号中获知成功。  
- **truncation**: 源代码未显示截断（max_steps），但多数环境有步数限制，此处未给出，视为不存在或不可直接用于奖励。  

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 空字典 `{}`，无可用字段 。  
- **forbidden_or_uncertain_info_fields**: 任何 info 字段均不可用（因为提供的信息为空）。终止原因也不能从 info 获取。  

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```
**允许使用**：  
- `obs`（当前观测）  
- `action`（当前动作）  
- `next_obs`（下一帧观测）  
- `info` 中明确允许的字段（实际为空，等于不可用）  
- `training_progress` 仅在 prompt 明确允许使用时可用，此处未允许，应禁止使用。  

**禁止使用**：  
- `original_reward`（被屏蔽的官方奖励）  
- 未声明的 `info` 字段（所有字段）  
- 未声明的 `obs` 切片（即不得依赖未记录的额外观测）  

## 7. 可用于奖励函数的信号
- **position**: `x_position`, `y_position`（可直接组成距离度量）  
- **velocity**: `x_velocity`, `y_velocity`（线速度）  
- **orientation**: `body_angle`, `angular_velocity`（角速度）  
- **contact**: `left_support_contact`, `right_support_contact`（双腿触地标志），可构造着陆条件  
- **action/engine**: 可以基于动作选择构建惩罚（如鼓励 no_engine，惩罚主引擎和姿态引擎）  
- **other**: 无  

## 8. 不确定或不可用的信号
- **官方奖励/任务终止标志**：被屏蔽，不可用  
- **绝对时间/步数**：无直接可用信号（training_progress 不允许使用）  
- **成功标志**：info 中无 success 或 failure 字段，不可用  
- **视口边界信息**：观测中无边界值，只能从位置推测越界但越界后 episode 已终止，故不能在奖励中直接使用边界信号  
- **风或其他扰动**：被省略，不可用  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D flying vehicle with two legs
  actuator_type: one main thruster (vertical), two orientation thrusters (torque)
  contact_structure: two-point support (left/right legs)
primary_objectives:
  - reach the central target landing pad (zero x,y relative position)
  - achieve stable contact with both legs, near-zero linear and angular velocity
  - keep body angle upright (close to 0)
secondary_objectives:
  - minimize total engine thrust usage (actuator effort)
  - arrive as fast as possible (implicit through fast approach)
main_failure_risks:
  - crash: body contacts ground or obstacle outside target pad
  - horizontal drift out of viewport
  - landing with single leg contact or high velocity, leading to instability
  - overshoot or oscillation around target, causing unnecessary fuel waste
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id**: `approach_goal`  
  **purpose**: 鼓励向目标着陆平台靠近，减少位置误差。  
  **why_required**: 前往目标是任务核心，距离奖是引导基础。  
  **usable_signals**: `x_position`, `y_position` （下一帧或当前帧）。  
  **risks**: 若权重过高，可能忽视软着陆条件，导致高速撞击；需要与速度/姿态惩罚协调。  

- **role_id**: `soft_landing_condition`  
  **purpose**: 检测着陆成功特征（双腿接触、低线速度、低角速度、竖直姿态），给予正向激励。  
  **why_required**: 任务要求稳定停靠，单纯到达不够，必须形成软着陆。  
  **usable_signals**: `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `angular_velocity`, `body_angle`。  
  **risks**: 条件设定过于严苛会导致延迟奖励，可能需配合稠密分量。  

- **role_id**: `stability_penalty`  
  **purpose**: 惩罚非竖直姿态和大角速度，鼓励飞行平稳。  
  **why_required**: 防止剧烈翻滚，便于着陆控制。  
  **usable_signals**: `body_angle`, `angular_velocity`。  
  **risks**: 与着陆时的必要姿态调整冲突；可考虑在接近目标后才加强。  

### 10.2 条件职责 conditional_roles
- **role_id**: `thrust_penalty`  
  **condition_to_use**: 当 agent 接近目标且速度已降至较低水平时逐渐启用，或在全程作为温和正则项。  
  **usable_signals**: `action`（是否使用主引擎或姿态引擎）。  
  **risks**: 过早惩罚会抑制探索；忽略则燃料浪费严重。建议随逼近程度动态调节权重。  

- **role_id**: `velocity_smoothing`  
  **condition_to_use**: 当需要更稳定轨迹时可加入，但不是必须。  
  **usable_signals**: `x_velocity`, `y_velocity` 的突变（需要历史信息，若无法获得则不可用）。  
  **risks**: 需要存储上一帧速度，增加复杂度；当前环境未提供历史，可能难以直接实现。  

### 10.3 慎用/禁用职责 avoid_roles
- **role_id**: `time_or_step_penalty`  
  **reason**: 无可用的步数或时间信号；`training_progress` 不允许使用，真实步数未暴露。  
  **forbidden_or_missing_signals**: 步数计数器。  

- **role_id**: `original_reward_mimic`  
  **reason**: 官方奖励被屏蔽，严禁尝试复现。  

- **role_id**: `safe_zone_boundary`  
  **reason**: 没有视口边界坐标，超出边界即终止，无法在 episode 内构建渐进惩罚。  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| `approach_goal` | `x_position`, `y_position` (from next_obs or obs) | — | `dense_state_signal` (e.g. -dist), `bounded_signal` | 可结合当前位置与下一帧位置计算趋近量 |
| `soft_landing_condition` | `left_support_contact`, `right_support_contact`, `x_velocity`, `y_velocity`, `body_angle`, `angular_velocity` (all from next_obs) | — | `gated_reward` (if all conditions met, large bonus) | 所有特征可从 next_obs 直接读取 |
| `stability_penalty` | `body_angle`, `angular_velocity` | — | `quadratic_penalty`, `abs_penalty` | 可直接加负平方或负绝对值 |
| `thrust_penalty` | `action` (Discrete index) | — | `action_cost` (e.g. penalty for action 1,2,3) | 动态权重可随与目标距离、速度调节 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 策略过早触发“crash”终止而无法获得正奖励 | 回报曲线持平且 episode 长度很短；多数终止时双腿未均接触 | 加强距离引导，增加 soft_landing 稀疏奖励，或减免早期动作惩罚 |
| 漂出水平边界 | 最终 x_position 数值超过视口边界（可从 episode 结束时观察估计） | 在接近边界附近增加基于横向位置的惩罚（结合 speed） |
| 长时间悬停而不着陆或燃料耗尽 | 回报不高，episode 步数很长且双腿未接触或接触不稳定 | 增加时间（步数）隐式压力，通过缓慢衰减奖励或加大 thrust_penalty |
| 着陆时只有单腿接触或倾斜 | 结束时 `left_support_contact` 与 `right_support_contact` 不一致，或 `body_angle` 较大 | 强化双腿同时接触的奖励条件，并加大 `body_angle` 惩罚 |
| 过度使用主引擎，省燃料目标未达成 | 动作统计中动作 2 使用频率高 | 适度调高 thrust_penalty 的权重，尤其是在低速靠近目标时 |
| 角速度过大导致旋转漂移 | 观察 `angular_velocity` 直方图 | 加强 angular_velocity 惩罚，或增加 anti‑spin