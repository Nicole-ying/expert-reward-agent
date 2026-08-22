# 匿名环境理解卡片

## 1. 任务目标  
本环境是一个二维飞行器/着陆器轨迹控制任务。智能体起始于视口上方中央，初始带有一个随机作用力。核心目标是 **让飞行器尽快到达中央目标垫，并实现稳定、安全的着陆停靠（soft landing）**。次目标包括：尽可能节省引擎推力（燃料效率）、保持接近竖直的姿态、避免水平漂移。不应混淆的目标：单纯的速度最小化或角度最小化必须与着陆成功与否关联，不能独立优化。

## 2. 任务类型选择  
selected_route_id: **navigation_goal_reaching**  
confidence: **high**  
reason: 任务的首要核心是到达指定目标垫，并在其上方精确着陆，属于典型的导航式目标到达任务，虽然附带了燃料与时间效率要求，但这些并非与到达同等权重的独立目标，因此不归为 multi_objective。

## 3. 观察空间 observation_space  
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32  
- 各维度说明：  
  - obs[0] **x_position**：飞行器相对于目标垫的水平坐标，reward_usable: true  
  - obs[1] **y_position**：飞行器相对于目标垫高度的垂直坐标，reward_usable: true  
  - obs[2] **x_velocity**：水平线速度，reward_usable: true  
  - obs[3] **y_velocity**：垂直线速度，reward_usable: true  
  - obs[4] **body_angle**：机体朝向角度，reward_usable: true  
  - obs[5] **angular_velocity**：角速度，reward_usable: true  
  - obs[6] **left_support_contact**：左支撑腿接触标志（0.0 或 1.0），reward_usable: true  
  - obs[7] **right_support_contact**：右支撑腿接触标志（0.0 或 1.0），reward_usable: true  

## 4. 动作空间 action_space  
- **type**: Discrete  
- **n**: 4  
- 动作及含义：  
  - action 0: **no_engine** – 不执行任何推力  
  - action 1: **left_orientation_engine** – 点火一个方向引擎，产生旋转力矩（通常用于逆时针旋转）  
  - action 2: **main_engine** – 点火主引擎（通常提供垂直向上的推力）  
  - action 3: **right_orientation_engine** – 点火另一个方向引擎，产生相反的旋转力矩  

## 5. step 与终止条件分析  

### 5.1 终止模式  
- **success-like termination**: `body_not_awake_or_settled` – 很可能表示机体已处于休眠状态且已稳定停靠在目标垫上（成功着陆）。  
- **failure-like termination**:  
  - `crash_or_body_contact` – 机体与地面或其他非垫部位发生猛烈碰撞（非腿部着陆接触）。  
  - `horizontal_position_outside_viewport` – 水平位置大幅偏离目标垫，飞出视野边界。  
- **ambiguous termination**: 无明确模糊终止，但 `body_not_awake_or_settled` 在无额外信息时可能包含非期望停靠（例如翻倒后静止），需结合腿接触与速度信号进一步确认。  
- **truncation**: step 源中未展示截断机制，假定无 episode 长度限制。

### 5.2 success/failure 信号可用性  
- **explicit_success_flag_available**: false（info 输出为空字典 `{}`）  
- **explicit_failure_flag_available**: false  
- **allowed_info_fields**: 仅允许使用空 info 字典，不可从中提取任何字段  
- **forbidden_or_uncertain_info_fields**: 所有未在 mask step source 中列出的字段（如 `"success"`, `"failure"`, `"termination_reason"` 等）均被禁止  

## 6. reward 函数接口契约  
函数签名：  
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：  
- `obs`（8 维 numpy 数组）  
- `action`（整数，0~3）  
- `next_obs`（8 维 numpy 数组）  
- `info` 中明确为空的字典，不可进行字段访问  
- `training_progress` 仅在 prompt 明确允许时使用，当前 **未** 获得许可  

**严格禁止使用**：  
- `original_reward`  
- 任何官方奖励复原值  
- 未在章节 3 或章节 7 中声明的 obs 维度  
- 任何未在 mask step source 中出现的 info 字段  

## 7. 可用于奖励函数的信号  
- **position**: `x_position`（obs[0]）, `y_position`（obs[1]）；均可直接度量与目标的距离  
- **velocity**: `x_velocity`（obs[2]）, `y_velocity`（obs[3]）；用于抑制高速及鼓励静止  
- **orientation**: `body_angle`（obs[4]）可惩罚过大倾斜，`angular_velocity`（obs[5]）可抑制过度旋转  
- **contact**: `left_support_contact`（obs[6]）, `right_support_contact`（obs[7]）；可用于识别着陆脚是否均接触目标垫  
- **action/engine**: 离散动作 `action`（0~3），可直接惩罚非零动作以节省燃料  
- **other**: 无其他额外可用信号  

## 8. 不确定或不可用的信号  
- 任何基于 `info` 的成功/失败标志不可用  
- 无法获得环境给出的显式燃料消耗量或剩余燃料量  
- 环境内部的重力、风力等参数未知，不得在奖励中显式假设  

## 9. 专家任务画像 expert_task_profile  
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 飞行器/着陆器
  actuator_type: 一个主引擎（向上推力） + 两个方向引擎（旋转力矩）
  contact_structure: 两条支撑腿，需双腿同时接触目标垫以稳定着陆
primary_objectives:
  - 抵达目标垫正上方并完成停靠（位置接近、速度归零、姿态竖直、双腿着地）
secondary_objectives:
  - 最小化引擎使用次数（燃料效率）
  - 尽量缩短飞抵时间
main_failure_risks:
  - 水平或垂直速度过高导致撞击损毁
  - 姿态过大导致翻倒或无效接触
  - 单腿着陆导致侧翻
  - 过早或过晚停止引擎导致不可控坠落
```

## 10. 奖励职责拆解 reward_role_decomposition  

### 10.1 主职责 mandatory_roles  
- **role_id: goal_distance_progress**  
  purpose: 鼓励飞行器向目标垫靠近（减小水平与垂直距离）  
  why_required: 任务核心是到达目标，必须持续提供到达进度信号  
  usable_signals: [x_position, y_position]  
  risks: 若权重过大，可能导致高速坠向目标垫  

- **role_id: soft_landing**  
  purpose: 要求在接近目标时速度趋于 0，形成轻柔着陆  
  why_required: 成功着陆不仅需要位置重合，更需稳定接触，否则碰撞会判定失败  
  usable_signals: [x_velocity, y_velocity, x_position, y_position]  
  risks: 可能被 agent 利用为悬停不降，需要与距离进度协同  

- **role_id: orientation_penalty**  
  purpose: 保持机体近乎竖直，避免过度倾斜  
  why_required: 倾斜时主推力的方向偏离垂直，降低控制效率，且倾斜着陆容易翻倒  
  usable_signals: [body_angle]  
  risks: 过于严厉可能抑制必要的姿态调整  

- **role_id: landing_contact_success**  
  purpose: 检测双腿是否均接触目标垫作为着陆成功的暗示  
  why_required: 环境未给出显式成功标志，只能通过接触与低速度组合推断成功，增强学习信号  
  usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity]  
  risks: 如果不结合低速度条件，可能误奖励滚动擦地行为  

- **role_id: fuel_efficiency**  
  purpose: 减少引擎使用次数  
  why_required: 任务明确要求尽可能少用引擎推力  
  usable_signals: [action]  
  risks: 若惩罚过强，可能导致不敢使用主引擎从而降落失败  

### 10.2 条件职责 conditional_roles  
- **role_id: angular_velocity_smoothing**  
  condition_to_use: 当角速度过大导致姿态震荡时，可辅助平滑  
  usable_signals: [angular_velocity]  
  risks: 可能抑制必要的旋转机动，需采用较小系数或仅在速度较高时激活  

- **role_id: time_efficiency_penalty**  
  condition_to_use: 若训练出现明显悬停拖延，可加入每步微小负值  
  usable_signals: none（纯步数惩罚）  
  risks: 添加固定步惩罚可能改变最优着陆策略，需要调参验证  

### 10.3 慎用/禁用职责 avoid_roles  
- **role_id: hard_success_flag**  
  reason: 环境未提供任何 info 成功标志，无法可靠识别真成功，强行推测会导致误奖励  
  forbidden_or_missing_signals: [info.success]  

- **role_id: explicit_episodic_time_limit**  
  reason: 未明确 episode 长度截断，且不在允许的 info 中，不应依赖环境时钟  
  forbidden_or_missing_signals: [episode_length, truncated]  

## 11. role_to_signal_mapping  
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_distance_progress | x_position, y_position | – | distance_to_target, bounded_signal | 可对水平、垂直距离分别或联合设计 |
| soft_landing | x_velocity, y_velocity, x_position, y_position | – | velocity_magnitude_penalty, proximity_gated_reward | 在距离目标较近时才要求低速，避免全程减速 |
| orientation_penalty | body_angle | – | quadratic_penalty, bounded_signal | 角度接近 0（竖直）时奖励 |
| landing_contact_success | left_support_contact, right_support_contact, x_velocity, y_velocity | – | binary_AND_gate, velocity_threshold | 双腿接触 + 速度极小 → 高奖励 |
| fuel_efficiency | action | – | action_cost_penalty | 对所有非零动作施加负代价 |
| angular_velocity_smoothing | angular_velocity | – | quadratic_penalty | 可选，视训练表现激活 |
| time_efficiency_penalty | – | episode_length | constant_step_penalty | 谨慎使用，可能干扰可控 |

## 12. 初始训练后应观察的 failure modes  
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 高速撞击目标垫 | y_velocity 在接触时仍为较大负值，或终止为 crash | 加强 soft_landing，或利用接触信号关闭对撞地误判 |
| 单腿着陆侧翻 | only one contact flag = 1，机体角度偏离较大 | 引入双腿同时接触奖励，提高 unbalance 惩罚 |
| 过远或不动（缺乏探索） | x_position, y_position 长期不变 | 检查动作熵，可能需增加探索噪声或少量目标位置激励 |
| 姿态振荡无法稳定 | angular_velocity 持续大幅波动，无法减速 | 加强 angular_velocity_smoothing，或微增方向引擎惩罚 |
| 无意义空转引擎浪费燃料 | 频繁使用左/右引擎但位置无明显变化 | 增加 action 代价系数，或对无进度情况额外惩罚 |
| 过早停止引擎导致失控坠落 | 接近目标时 action=0 突然坠地 | 优化 soft_landing 的 proximity_gated 阈值，确保接近时才激活滞空鼓励 |