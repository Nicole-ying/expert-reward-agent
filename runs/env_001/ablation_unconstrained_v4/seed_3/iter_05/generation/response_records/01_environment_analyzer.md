# Response Record

# 匿名环境理解卡片

## 1. 任务目标
该任务是 2D 飞行器/着陆器轨迹优化问题。  
**主目标**：从上方初始位置出发，以最快速度安全降落到视口正中的目标着陆垫上，最终在垫上保持稳定静止（settle）。  
**次要目标**：在此过程中尽量减少引擎推力使用（燃料经济性），维持姿态稳定（防止倾倒或侧翻）。  
**避免混淆**：不是单纯的速度控制或姿态平衡任务；到达目标垫并稳固着陆是唯一的核心成功条件，燃料节省和姿态稳定只是加分项。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是“移动到指定目标位置并稳定停止”，存在明确的到达目标（目标垫），到达后要求保持静止。虽然存在姿态稳定和燃料节省的次要目标，但它们服务于主目标或作为附加追求，不会从根本上改变任务性质。因此适合归为导航目标到达类。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（每个元素在连续区间取值）
- obs[0]: x_position，水平相对目标垫的坐标，reward_usable: true
- obs[1]: y_position，垂直相对目标垫高度的坐标，reward_usable: true
- obs[2]: x_velocity，水平速度，reward_usable: true
- obs[3]: y_velocity，垂直速度，reward_usable: true
- obs[4]: body_angle，机体朝向角，reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左侧支撑触点标志（0.0 / 1.0），reward_usable: true
- obs[7]: right_support_contact，右侧支撑触点标志（0.0 / 1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine，不点火
- action 1: left_orientation_engine，左姿态调节引擎点火
- action 2: main_engine，主引擎点火（可能向下产生推力）
- action 3: right_orientation_engine，右姿态调节引擎点火

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（可能意味着稳定着陆）可能是唯一与成功相关的终止条件，但代码中与其他失败条件合并为一根 `terminated` 标志，无法直接区分。
- failure-like termination: crash_or_body_contact、horizontal_position_outside_viewport 通常为失败。
- ambiguous termination: body_not_awake_or_settled 的内涵既可能是成功（平稳停靠）也可能是失败（摔落后不动）。
- truncation: 未提供（无时间上限标记）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 固定为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，特别是 `success`、`failure`、`termination_reason` 等不存在。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action（当前动作）
- next_obs（下一帧观测）
- training_progress 仅在明确允许时使用（当前环境中未提示可用，默认不使用）

禁止使用：
- original_reward（被官方掩码）
- info（为空）
- 任何未声明的 obs 切片
- 任何环境内部变量

## 7. 可用于奖励函数的信号
- position: x_position (obs[0]), y_position (obs[1]) 以及 next_obs 中的对应值
- velocity: x_velocity (obs[2]), y_velocity (obs[3])，可构建速度范数
- orientation: body_angle (obs[4]), angular_velocity (obs[5])
- contact: left_support_contact (obs[6]), right_support_contact (obs[7])
- action/engine: action 取值 0–3，可用来惩罚推力使用频率，但不能获取具体推力数值

## 8. 不确定或不可用的信号
- 官方奖励值：不可用，必须忽略。
- 终止原因分解：不可用，无法区分是成功着陆还是失败。
- 燃料残余量或具体冲量值：不可用。
- 目标垫半径或接触面积参数：未提供。
- 外部风或扰动信号：不可用。
- 任何 info 键值：不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: single_rigid_body_with_legs_or_contacts
  actuator_type: lateral_orientation_engines + main_engine
  contact_structure: two_foot_support (left/right legs)
primary_objectives:
  - reach_target_pad: 将 x 和 y 坐标驱动至目标垫中心附近
  - stabilize_on_pad: 在垫上速度趋于零且保持正确姿态（角度近 0）
  - avoid_crash: 防止 hard contact 或飞出视口
secondary_objectives:
  - fuel_efficiency: 减少引擎使用次数
  - orientation_smoothness: 减少大幅角速度波动
main_failure_risks:
  - 直接撞向目标垫（垂直速度过大导致 crash）
  - 横向移动超出视口
  - 姿态过度倾斜导致在垫上侧翻
  - 过度使用引擎导致燃料耗尽（隐含）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: approach_progress  
  purpose: 鼓励智能体向目标垫移动，缩短距离。  
  why_required: 不指定此角色，智能体可能没有趋近目标的动机。  
  usable_signals: [x_position, y_position, next_x_position, next_y_position]（可构成距离减少）  
  risks: 若只奖励靠近不惩罚速度，可能导致撞击。

- role_id: soft_landing  
  purpose: 在目标垫附近时，要求低速、姿态端正，且两侧触地。  
  why_required: 需要将“到达区域”与“安全停靠”绑定，否则智能体可能仅掠过目标。  
  usable_signals: [x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact, 目标垫相对位置]  
  risks: 过早施加停靠惩罚会阻碍探索；可在距目标垫一定范围内才激活。

### 10.2 条件职责 conditional_roles
- role_id: fuel_penalty  
  condition_to_use: 当动作使用了引擎（action != 0），且训练需要优化燃料时加入。  
  usable_signals: [action]  
  risks: 权重过高会导致智能体完全不使用引擎，无法飞行。

- role_id: orientation_penalty  
  condition_to_use: 姿态偏离竖直方向较大时，可能用于预防侧翻；可与 soft_landing 合并。  
  usable_signals: [body_angle, angular_velocity]  
  risks: 单独使用会抑制必要的姿态调整动作。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: terminal_success_reward  
  reason: 无明确的成功标志，不能依据 terminated 给予大额奖励，因为无法区分成功/失败。  
  forbidden_or_missing_signals: [explicit success/failure flag]

- role_id: crash_penalty_based_on_info  
  reason: info 为空，不能从 info 读取碰撞详情。  
  forbidden_or_missing_signals: [crash_info]

- role_id: time_penalty  
  reason: 虽然有“尽快到达”的次要目标，但当前环境无步数限制且未提供时间信号，也不应为了速度惩罚牺牲安全探索。可以暂时不使用，或仅在后续 fine‑tune 阶段考虑。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| approach_progress | x_position, y_position, next_x_position, next_y_position | 目标垫绝对坐标（需相对） | distance_change, dense_state_signal | 使用 `-(dist_next - dist_now)` 鼓励靠近 |
| soft_landing | x_velocity, y_velocity, body_angle, angular_velocity, left_contact, right_contact, distance_to_pad | 精确 contact 力度 | bounded_signal, quadratic_penalty | 仅在 distance < threshold 时激活，惩罚速度、角度、接触不对称 |
| fuel_penalty | action (0-3) | 单次推力大小 | constant_penalty_per_nonzero_action | 仅当 action != 0 时给予负奖励 |
| orientation_penalty | body_angle, angular_velocity | 无 | quadratic_penalty | 可与 soft_landing 中角度约束共享 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 只悬停不向目标垫移动 | x,y 距离未减小，速度接近零 | 加强 approach_progress 的权重，或降低燃料惩罚 |
| 高速撞向目标垫后终止 | y 速度较大时终止，但距离目标很近 | 激活 soft_landing 的速度惩罚，可结合速度阈值 |
| 在目标垫上方左右摆动 | x 坐标反复过冲，角速度大 | 加入姿态平滑惩罚、接近目标时减小引擎奖励 |
| 仅使用单侧引擎造成旋转坠落 | 角度逐渐发散，最后 crash | 增加方向与角速度惩罚，且对不平衡 contact 给予代价 |
| 完全不使用引擎 | 距离不减少，reward 趋平 | 减少燃料惩罚或改为只在成功着陆后给予 fuel bonus |
