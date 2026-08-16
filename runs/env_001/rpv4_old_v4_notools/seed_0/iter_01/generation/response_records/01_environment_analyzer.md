# Response Record

# 匿名环境理解卡片

## 1. 任务目标
智能体控制一个2D飞行器（类似着陆器），初始位于视野上方并带有随机外力。核心目标是**快速、平稳地降落在中央目标着陆垫上并稳定下来**，同时尽可能**减少发动机推力使用**。智能体需要学习接近目标、降低速度、保持稳定姿态、并实现安全接触。注意：不应混淆为单纯的能量节省任务——到达目标垫是首要条件，速度和姿态的平滑性是实现安全着陆的手段，而推力节省是重要但次要的优化目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是到达并停靠于中央目标垫，有明确的空间目标（goal pad）。次要目标（速度、姿态、燃料）均服务于安全着陆或性能优化，未构成权重相当的多目标冲突。因此属于导航目标到达族。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (实际为连续值，左/右支撑为 0.0 或 1.0)
- 各维度含义：
  - obs[0] (x_position): 相对于目标着陆垫的水平坐标，reward_usable: true  
  - obs[1] (y_position): 相对于着陆垫高度的垂直坐标，reward_usable: true  
  - obs[2] (x_velocity): 水平线速度，reward_usable: true  
  - obs[3] (y_velocity): 垂直线速度，reward_usable: true  
  - obs[4] (body_angle): 机体角度（方向），reward_usable: true  
  - obs[5] (angular_velocity): 角速度，reward_usable: true  
  - obs[6] (left_support_contact): 左侧支撑接触标志（1.0表示接触，0.0表示未接触），reward_usable: true  
  - obs[7] (right_support_contact): 右侧支撑接触标志，reward_usable: true

所有观测维度均可在自定义奖励中安全使用。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - action 0: no_engine – 什么都不做，滑翔/惯性飞行
  - action 1: left_orientation_engine – 点燃左侧姿态发动机（产生逆时针或顺时针旋转力矩）
  - action 2: main_engine – 点燃主发动机（提供向机头方向的推力，通常向上或减速）
  - action 3: right_orientation_engine – 点燃右侧姿态发动机（与左侧相反的旋转力矩）

注意：该飞行器没有横向直接推进器，水平控制主要通过倾斜机身+主发动机实现。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled 中的 “settled” 部分 —— 推测为飞行器稳定停靠在目标着陆垫上（满足速度、角度阈值且支撑接触）
- failure-like termination: crash_or_body_contact（当不属于成功着陆时的身体/地面碰撞）和 horizontal_position_outside_viewport
- ambiguous termination: body_not_awake_or_settled 内部 —— 由于任务描述未区分“settled”与“not awake”，但根据任务目标“reach and settle”可推断出成功情况；需结合接触标志、位置接近度进一步解析
- truncation: 无（step 返回的 truncated 恒为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，终止后无明确标志）
- explicit_failure_flag_available: false 
- allowed_info_fields: 无（info 为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均未提供，不可依赖。

成功/失败只能通过下一状态的观测（next_obs）与 done 组合推断：例如，若终止时左右支撑接触均为 1，且 x、y 位置接近 0，速度很小，角度接近 0（或垂直），则可以假定为成功着陆。

## 6. reward 函数接口契约
函数签名（由外部调用者强制遵循）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs: 前一步的 8 维观测（可忽略，若有需要）
- action: 实际执行的动作编号
- next_obs: 执行动作后的 8 维观测（绝大多数奖励信号应来源于此）
- info: 仅限空字典，无任何可用信息
- training_progress: 任务 prompt 未允许使用，因此必须忽略（可视为缺省 0.0，但不应根据它设计奖励）

禁止使用：
- original_reward（被屏蔽的官方奖励）
- 任何未声明的 info 字段
- 任何未在 obs 中明确给出的传感器，如额外的碰撞深度、燃料存量等

## 7. 可用于奖励函数的信号
- 位置信号：
  - next_obs[0] (x_position): 相对于目标着陆垫的水平偏差
  - next_obs[1] (y_position): 相对于着陆垫高度的垂直偏差（越高越正？）
- 速度信号：
  - next_obs[2] (x_velocity): 水平速度（靠近着陆垫时应接近 0）
  - next_obs[3] (y_velocity): 垂直速度（靠近着陆垫时应接近 0 或一个小负值以表示软下降）
- 姿态信号：
  - next_obs[4] (body_angle): 机体角度，用于维持稳定姿态（接近 0 表示竖直）
  - next_obs[5] (angular_velocity): 角速度，用于惩罚大幅旋转
- 接触信号：
  - next_obs[6] (left_support_contact): 左侧支撑接触 (0/1)
  - next_obs[7] (right_support_contact): 右侧支撑接触 (0/1)
  可用于判断着陆状态、两侧接触平衡、软/硬着陆
- 动作/发动机信号：
  - action: 离散动作，可用于惩罚推力使用（action 0 为无推力，action 1/3 为姿态推力，action 2 为主发动机推力）
- 其他衍生物：
  - 距离到目标：sqrt(x_position² + y_position²)
  - 接近速度/方向
  - 着陆瞬间的状态组合（位置、速度、角度）

## 8. 不确定或不可用的信号
- 明确的“成功着陆”标记（不存在于 info 中）
- 燃料剩余量（观测中未提供）
- 着陆垫的精确接触力/接触稳态信息（仅有接触标志）
- 物理参数如质量、惯性等（不可用）
- 外部风/随机力（被屏蔽，仅通过初始随机力隐含）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 单一刚体飞行器（带两个支撑足/接触点）
  actuator_type: 离散推力发动机（主推进 + 姿态旋转）
  contact_structure: 两点着陆支撑，通过接触标志反馈
primary_objectives:
  - 到达并停在目标着陆垫（x ≈ 0, y ≈ 0 且支撑接触）
secondary_objectives:
  - 最小化发动机使用次数（尤其主发动机）
  - 保持机体接近竖直方向（body_angle ≈ 0）
  - 着陆时垂直速度和水平速度接近零
  - 尽快完成着陆
main_failure_risks:
  - 硬着陆（高速垂直或水平接触导致 crash）
  - 倾斜着陆（大角度接触导致不稳定或继续滑落）
  - 滞空盘旋过久浪费燃料（实际无燃料限制，但动作浪费）
  - 水平飘移出界
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_proximity  
  purpose: 驱动智能体靠近目标着陆垫  
  why_required: 这是任务的首要目标，没有接近动作就不可能完成着陆  
  usable_signals: [next_obs[0], next_obs[1]] （可用 `-sqrt(x²+y²)` 或分段奖励）  
  risks: 若仅用接近奖励，可能诱使智能体高速撞向目标，需与速度/姿态约束结合

- role_id: soft_landing_success  
  purpose: 在成功软着陆时给予大幅正向奖励，指示任务完成  
  why_required: 任务明确要求“settle at the target pad”，稀疏的成功信号可从根本上引导最终行为  
  usable_signals: [done, next_obs[6], next_obs[7], next_obs[0], next_obs[1], next_obs[3], next_obs[4]] （通过判定 done 且两腿接触、x,y≈0、垂直速度小、角度接近0等）  
  risks: 如果设置的阈值过严，可能导致探索不到成功状态；需适当宽松

- role_id: safe_orientation  
  purpose: 惩罚过大的机体角度和角速度，保持稳定姿态  
  why_required: 为防止翻滚或非竖直着陆（可能导致不平衡或硬着陆），姿态控制是必须的安全约束  
  usable_signals: [next_obs[4], next_obs[5]]  
  risks: 过强的角度惩罚可能限制智能体必要的倾斜动作，影响水平位置调整

### 10.2 条件职责 conditional_roles
- role_id: velocity_damping_near_pad  
  purpose: 当靠近目标着陆垫时，鼓励降低水平和垂直速度，以实现软接触  
  condition_to_use: 仅当 `sqrt(next_obs[0]**2 + next_obs[1]**2)` 小于某个阈值（例如 0.5）且未终止时  
  usable_signals: [next_obs[2], next_obs[3], next_obs[0], next_obs[1]]  
  risks: 如何正确设定阈值并平衡与接近奖励；若过早启用可能拖慢下降

- role_id: action_efficiency  
  purpose: 惩罚不必要的发动机使用，节省“燃料”（尽管无燃料计量，但可视为时间/能量优化）  
  condition_to_use: 始终可应用，但建议在靠近目标时权重降低，避免因为追求零推力而错失精确调整  
  usable_signals: [action] （例如 action==0 奖励小正值，action==2 惩罚较大）  
  risks: 若权重过高，智能体可能学不会使用姿态发动机调整水平位置，导致无法正确着陆

### 10.3 慎用/禁用职责 avoid_roles
- role_id: contact_balance  
  reason: 没有提供“两侧接触力”信号，只有二元接触标志，无法精确计算平衡；但可通过“两侧同时接触”作为成功条件的一部分，无需单独奖励  
  forbidden_or_missing_signals: [缺失连续接触力或扭矩]

- role_id: fuel_penalty_linear  
  reason: 环境中没有燃料剩余状态，无法实现基于燃料的惩罚；但可用动作计数替代，已含在 action_efficiency 中  
  forbidden_or_missing_signals: [无燃料测量]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_proximity | next_obs[0], next_obs[1] | — | negative_euclidean, bounded_signal (如 reward=exp(-d²/σ)) | 直接欧氏距离负值，或采用二次负奖励 |
| soft_landing_success | done, next_obs[6], next_obs[7], next_obs[0], next_obs[1], next_obs[3], next_obs[4] | — | sparse_event | 在终止且条件满足时给予大奖励（如 +100），需小心阈值 |
| safe_orientation | next_obs[4], next_obs[5] | — | quadratic_penalty, absolute_penalty | 可用 -w1*angle² - w2*angvel² |
| velocity_damping_near_pad | next_obs[2], next_obs[3] (仅靠近时) | 需要定义“靠近”的指示器 | velocity_penalty_scaled_by_proximity | 如 -|vx|*g(d) |
| action_efficiency | action | — | discrete_penalty | action==2 罚得多，action==1/3 罚得少，action==0 奖励小正值 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 智能体只盘旋不下降 | 长期 y_position 居高不下，done 频率低 | 强化 goal_proximity 关于 y 的权重，或加入对长期不下降的惩罚 |
| 高速撞向地面 (hard landing) | done 时 y_velocity 大负值，crash 比例高 | 增强靠近垫子时的速度阻尼，或调整安全着陆的时间奖励 |
| 过度旋转或不稳定 | body_angle 或 angular_velocity 持续增大，未着陆就翻滚 | 提高 safe_orientation 权重，加入角速度二阶惩罚 |
| 水平飘移出界 | 多次 horizontal_position_outside_viewport | 加强 x_position 的零偏差惩罚，或限制水平速度阈值 |
| 停滞在垫子上方但最终漂走 | 接近目标但未触发 settled，然后横向移动出界 | 可能成功条件阈值不准确，或需要添加对“在垫上方停留”的持续奖励 |
| 对称性能量问题（只用主发动机不加姿态） | 角度误差大，无法调整水平位置，最终撞地或出界 | 需要增强 action_efficiency 与安全着陆条件的平衡，必要时在远处放宽引擎惩罚 |
