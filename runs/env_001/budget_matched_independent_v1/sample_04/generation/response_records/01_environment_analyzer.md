# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本体从屏幕顶部中心区域出发，受到初始随机力。核心任务是**到达并稳定停在屏幕中央的目标平台上**，同时满足以下次目标：  
- **尽快完成**（步数效率）  
- **尽量少用引擎推力**（燃料经济性）  
- 保持本体姿态稳定、低速接触  
该任务不追求多目标平等权衡，到达与停稳是唯一主线，其余为辅助优化。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是到达并停稳在一个明确的位置目标（目标垫），使用引擎控制运动轨迹，属于经典的导航到达类问题。附属目标（速度、燃料、姿态）优先级低于到达目标，不构成多目标冲突。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64 (推测，未明确指定，但符合大多数 Box 环境)  
各维含义及奖励可用性：

- obs[0] (x_position): 本体水平坐标，相对于目标垫中心（可能已居中化为相对值）。reward_usable: true  
- obs[1] (y_position): 本体垂直坐标，相对于垫面高度（即本体脚踝或机身相对于垫面的垂直距离）。reward_usable: true  
- obs[2] (x_velocity): 水平线速度。reward_usable: true  
- obs[3] (y_velocity): 垂直线速度。reward_usable: true  
- obs[4] (body_angle): 本体姿态角（弧度或归一化角度，通常 0 为竖直）。reward_usable: true  
- obs[5] (angular_velocity): 角速度。reward_usable: true  
- obs[6] (left_support_contact): 左支撑腿是否接触地面/垫（1.0 接触, 0.0 未接触）。reward_usable: true  
- obs[7] (right_support_contact): 右支撑腿接触标志。reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
各动作含义：  
- action 0 (no_engine): 不点火，无推力。  
- action 1 (left_orientation_engine): 点燃左姿态引擎（产生侧向推力，可能产生旋转或侧移）。  
- action 2 (main_engine): 点燃主引擎（通常提供向上推力）。  
- action 3 (right_orientation_engine): 点燃右姿态引擎（与左引擎对称）。

## 5. step 与终止条件分析
### 5.1 终止模式
依据 `terminated` 信号的三个来源：
- **crash_or_body_contact**：本体与地面/物体发生碰撞，或本体与垫接触。该条件本身**无法区分成功着陆与坠毁**（成功着陆也必然伴随接触）。属于 **ambiguous termination**。
- **horizontal_position_outside_viewport**：本体水平位置逸出屏幕边界。**明确失败**。
- **body_not_awake_or_settled**：本体不再被物理引擎唤醒，或已稳定（无速度、无转动）。可能意味着成功停稳，也可能是因物理错误停止。按常理推断，该终止条件常用于标记“着陆完成”。属于 **success-like termination**（但不能 100% 确定）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 固定为 {}）  
- forbidden_or_uncertain_info_fields: 全部禁止，因 info 空

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs / action / next_obs
- 上述三个参数的全部字段（按前述定义）
- training_progress 仅在 prompt 明确允许时使用（当前未允许，禁止使用）

禁止使用：
- original_reward
- official_reward
- info（info 为空，即使有也不允许使用）
- 任何未声明的观测切片或内部变量

## 7. 可用于奖励函数的信号
- position: x_position, y_position（直接可读，可计算到目标垫中心的距离）
- velocity: x_velocity, y_velocity
- orientation: body_angle, angular_velocity
- contact: left_support_contact, right_support_contact（二值化标志）
- action/engine: 当前动作（可推算哪个引擎被点燃）
- other: 无其他额外信息

## 8. 不确定或不可用的信号
- 成功/失败显式标志：无
- 燃料消耗率/剩余燃料：无
- 目标垫的精确世界坐标：x_position、y_position 已经是相对值，故世界坐标不可知，但任务不需要
- 碰撞类型（crash vs soft contact）：无法从观测直接区分
- 任何 info 字典中的信号：空

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D vehicle-like lander (类似月球着陆器)
  actuator_type: 3 个推力器：主引擎（向上推力）、两个侧向姿态引擎
  contact_structure: 两条支撑腿，接触由左右接触标志给出
primary_objectives:
  - 到达并稳定停在中央目标垫上（位置收敛、速度趋零、姿态水平）
secondary_objectives:
  - 最小化步数（快速完成）
  - 最小化引擎使用（节省燃料）
main_failure_risks:
  - 高速坠地导致 crash
  - 水平漂移超出视口
  - 无法减速导致反复弹跳或无法 settled
  - 姿态失控、旋转碰撞
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 驱动本体靠近目标垫（水平居中、垂直下降）  
  why_required: 任务不提供到达目标的内生信号，必须用位置构造趋近奖励  
  usable_signals: [x_position, y_position]  
  risks: 若仅用距离，可能鼓励高速冲向目标，需配合减速约束

- role_id: soft_landing_velocity  
  purpose: 惩罚过大线速度，尤其接近垫面时强调减速  
  why_required: 单纯趋近会导致硬着陆，必须减缓速度以满足“稳定接触”要求  
  usable_signals: [x_velocity, y_velocity]  
  risks: 过早惩罚可能抑制探索，应与趋近奖励平衡

- role_id: orientation_stability  
  purpose: 保持本体姿态接近竖直（body_angle ≈ 0）  
  why_required: 侧向倾斜容易导致接触失败或侧翻，对成功停放至关重要  
  usable_signals: [body_angle]  
  risks: 轻微倾斜可能被过度惩罚，需要使用平滑代价

- role_id: contact_encouragement  
  purpose: 鼓励支撑腿接触目标垫  
  why_required: 成功停放的必要物理条件，且该信号可以在未终止前提供中期反馈  
  usable_signals: [left_support_contact, right_support_contact]  
  risks: 若奖励过高可能诱导刷接触点行为，需结合位置/速度约束

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency  
  condition_to_use: 当本体已靠近目标垫且速度较小时，才逐渐引入引擎使用惩罚；或在整个训练后期通过 training_progress 缩放  
  usable_signals: [action == 1,2,3]  
  risks: 过早引入会导致 agent 不敢使用引擎，无法学习精确控制

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_reward  
  reason: 没有显式成功标志，无法区分终止类型，强行给予大正奖励风险极高  
  forbidden_or_missing_signals: [explicit_success_flag]

- role_id: progress_rew_from_original_reward  
  reason: 原始奖励被屏蔽，契约禁止使用  
  forbidden_or_missing_signals: [original_reward]

- role_id: survival_bonus  
  reason: 任务不是平衡/存活类，活得更久没有内在价值，可能推迟到达  
  forbidden_or_missing_signals: 无合适的存活信号

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | x_position, y_position | 无 | negative_distance, bounded_signal | 可计算欧几里得距离，或用对 x/y 独立线性惩罚 |
| soft_landing_velocity | x_velocity, y_velocity | 无 | quadratic_penalty, bounded_signal | 速度接近 0 奖励最高，尤其垂直速度 |
| orientation_stability | body_angle | 无 | quadratic_penalty | angle 可平方或取绝对值惩罚 |
| contact_encouragement | left_support_contact, right_support_contact | 无 | binary_bonus | 单个接触给正奖励，双接触加倍 |
| fuel_efficiency (conditional) | action | 无 | action_mask_penalty | main engine (act=2) 可罚更多；姿态引擎较轻 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 悬停不降 | y_position 不减小，y_velocity 长期接近 0，episode 长度很长 | 增强接近奖励中 y 方向的权重，或降低速度惩罚的阈值 |
| 高速俯冲 | y_velocity 过大负值，crash 终止频繁 | 增加 soft_landing_velocity 权重，尤其对负 y 速度更严厉 |
| 横向漂出视口 | x_position 发散，最终因边界终止 | 强化 x 位置趋近奖励，或加入 x_velocity 罚项以抑制横向移动 |
| 反复弹跳无法 settled | 接触标志频繁跳变，y_velocity 正负交替 | 提高接触
