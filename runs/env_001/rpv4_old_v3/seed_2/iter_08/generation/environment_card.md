# 匿名环境理解卡片

## 1. 任务目标
任务主目标是控制 2D 飞行器从视口顶部中央附近开始，尽快到达并稳定降落在中央目标平台上。降落必须安全：接近目标点、速度极低、姿态稳定、身体与平台温和接触。次要目标包括尽量减少发动机使用（燃料经济），隐含了速度‑燃料之间的权衡。不应混淆为纯粹的存活平衡问题或连续地形行走问题，其核心是导航到达与软着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 主要目标是到达指定目标位置（着陆 pad），并进行安全接触与稳定停留。快速到达与节省燃料为附属优化项，并非权重相当的多目标。动力学强调接近目标、减速、姿态调整、软着陆，完全符合导航到达类型。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（基于典型实现，但未显式声明，可认为是浮点）  
- 各维度含义及奖励可用性：  
  - obs[0]: x_position（水平相对目标距离），reward_usable: true  
  - obs[1]: y_position（垂直相对目标高度），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体角度），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支腿接触标志，1/0），reward_usable: true  
  - obs[7]: right_support_contact（右支腿接触标志，1/0），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：  
  - action 0: no_engine（无发动机推力）  
  - action 1: left_orientation_engine（左姿态发动机点火）  
  - action 2: main_engine（主发动机点火，提供向上推力为主）  
  - action 3: right_orientation_engine（右姿态发动机点火）  

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 身体处于非活跃或已稳定（body_not_awake_or_settled），同时未触发 crash 或出界。此条件通常意味着着陆成功并静止。  
- failure‑like termination:  
  - crash_or_body_contact（极可能是坠毁性接触，如机身撞地、速度过高时的接触）。  
  - horizontal_position_outside_viewport（水平飞出视口）。  
- ambiguous termination: body_not_awake_or_settled 单独无法区分成败，需结合其他条件，但代码中终止判断为“crash_or_body_contact or out of viewport or not awake/settled”，一旦满足任一即终止，因此成功着陆必然伴随 not_awake/settled 且无 crash/出界。  
- truncation: 未提及时间限制，但可能存在最大步长截断（无显式声明）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（step 返回 info = {}）  
- forbidden_or_uncertain_info_fields: 所有 info 字段不可用，包括任何可能存在的 “success”“failure”“termination_reason”  

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用的信号：
- obs 全部 8 维  
- action（0~3 离散值）  
- next_obs（同 obs 结构）  
- training_progress（本次 prompt 未明确允许使用，建议忽略）  

严格禁止使用的信号：
- original_reward（被屏蔽，必须忽略）  
- info（任何字段，包括可能存在的 success、failure 标志）  
- 任何未声明的 obs 切片或全局变量  

## 7. 可用于奖励函数的信号
- 位置偏差：obs[0]（x 距离），obs[1]（y 距离）  
- 速度：obs[2]（vx），obs[3]（vy）  
- 姿态：obs[4]（角度），obs[5]（角速度）  
- 接触：obs[6]（左支腿接触），obs[7]（右支腿接触）  
- 动作：action ∈ {0,1,2,3}（可用于燃料惩罚或鼓励特定行为）  
- next_obs：可提供下一时刻状态，但应避免依赖未来的接触标志等，可做保守使用（如用于差分速度奖励）

## 8. 不确定或不可用的信号
- 明确的“坠毁”标签或大惩罚信号：无直接标志，只能通过位置越界、速度过高、角度过大等间接推断  
- 成功着陆标志：无  
- 任务完成度（progress）：不可直接获得  
- 剩余时间或步数：不可用  
- 原环境奖励：官方奖励被隐藏，不可在自定义奖励中引用  

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D 刚体飞行器
  actuator_type: |-
    主发动机（垂直推力为主）+ 两个姿态发动机（提供旋转力矩）
  contact_structure: 左右两个支腿，各自有独立的接触检测
primary_objectives:
  - 安全降落到目标平台（位置靠近 x≈0,y≈0，速度极低，角度趋于0）
  - 避免坠毁、飞出视口或无法稳定
secondary_objectives:
  - 最小化发动机使用（节省燃料）
  - 快速完成降落（隐含在燃料与速度的权衡中）
main_failure_risks:
  - 下降速度过大导致硬着陆（crash）
  - 姿态失控，机身撞击地面或平台边缘
  - 水平漂移出视口
  - 长时间振荡无法稳定着陆
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: position_approach  
  purpose: 引导飞行器朝目标 pad 移动，减小 x 和 y 的偏移。  
  why_required: 这是到达任务的基本驱动，没有它 agent 不会向目标靠近。  
  usable_signals: [obs[0], obs[1]]（也可用 next_obs 的同位置）  
  risks: 如果权重过高可能忽视姿态和速度，导致快速但猛撞。  

- role_id: velocity_damping  
  purpose: 要求接触前速度极低，尤其在垂直方向。  
  why_required: 软着陆的核心条件，速度过高将被视为失败。  
  usable_signals: [obs[2], obs[3]]，以及它们的平方或绝对值。  
  risks: 太强的阻尼会使 agent 行动过于保守，下降太慢。  

- role_id: orientation_stabilization  
  purpose: 保持机体接近竖直（角度趋向0），抑制角速度。  
  why_required: 过大倾斜会导致不安全接触或支腿无法同时触地。  
  usable_signals: [obs[4], obs[5]]  
  risks: 过度惩罚角速度可能抑制必要的姿态调整。  

- role_id: fuel_penalty  
  purpose: 对使用发动机的动作施加负奖励，鼓励低推力策略。  
  why_required: 实现“尽可能少用引擎”的次要目标，同时避免无意义的推力浪费。  
  usable_signals: [action]  
  risks: 过重可能导致 agent 不敢点火，无法减速或纠正姿态。  

- role_id: crash_avoidance  
  purpose: 惩罚飞行器进入危险状态（极速、极端姿态、远离 pad）。  
  why_required: 缺乏明确坠毁信号时，必须通过状态约束避免灾难性行为。  
  usable_signals: [obs[0], obs[1], obs[2], obs[3], obs[4]] 的阈值组合。  
  risks: 阈值设定不当可能产生错误惩罚或过于松驰。  

### 10.2 条件职责 conditional_roles
- role_id: soft_contact_encouragement  
  condition_to_use: 当飞行器已非常接近目标（|x|、|y| 小于阈值）且速度较低时，鼓励左右支腿同时接触。  
  usable_signals: [obs[6], obs[7]]  
  risks: 过早鼓励接触可能导致 agent 在远离目标时做无用触地尝试；与 position_approach 可能冲突。  

### 10.3 慎用/禁用职责 avoid_roles
- role_id: success_bonus  
  reason: 无显式成功标志，无法判断任务完成时刻，强行使用会导致稀疏且无法定义。  
  forbidden_or_missing_signals: [success_flag (info 中不存在)]  

- role_id: time_efficiency_penalty  
  reason: 无法获取 episode 时间或剩余步数，训练进度信号 training_progress 未被明确允许使用。  
  forbidden_or_missing_signals: [time_step, episode_length]  

- role_id: progress_speed_bonus  
  reason: 没有明确的进度度量（如已移动距离占全程的比例）；直接鼓励高速将损害安全要求。  
  forbidden_or_missing_signals: [normalized_progress]  

- role_id: energy_efficiency（单独再列）  
  reason: 已由 fuel_penalty 涵盖，且缺少精确能耗计算公式。  
  forbidden_or_missing_signals: [thrust_magnitude, fuel_consumption]  

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| position_approach | obs[0], obs[1] | – | negative\_l2\_norm, negative\_manhattan | 可分段设置不同权重，远距离可更宽松 |
| velocity_damping | obs[2], obs[3] | – | negative\_squared, negative\_abs | y 速度通常需要更大权重 |
| orientation_stabilization | obs[4], obs[5] | – | negative\_abs_angle, quadratic\_penalty | 角速度也应衰减 |
| fuel_penalty | action | – | step\_constant\_penalty, per\_action\_cost | 不同动作可有不同代价（主引擎更贵） |
| crash_avoidance | obs[0], obs[1], obs[2], obs[3], obs[4] | crash\_flag | threshold\_penalty, large\_constant\_penalty | 需要设定安全边界并给予重罚 |
| soft_contact_encouragement | obs[6], obs[7] (以及 obs[0], obs[1], obs[2], obs[3] 作为条件) | – | binary\_bonus, conditional\_reward | 建议仅在近目标低速时激活 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 过度振荡，无法稳定着陆 | episode 长度过长；位置/速度/角度持续波动 | 增强 velocity_damping 和 angle 惩罚，调整 gain |
| 高速直冲 pad，导致 crash | 最终几步速度极大，奖励曲线骤降 | 加大 crash_avoidance 阈值惩罚，提前施加速度阻尼 |
| 不使用主引擎，仅靠姿态引擎漂移 | action 统计中主引擎使用率极低，下降过慢 | 减轻 fuel_penalty 对主引擎的惩罚，或引入时间衰减 |
| 忽略姿态，机身横躺接触 | 最终 body_angle 很大，仅单腿接触 | 强化 orientation_stabilization 和 soft_contact 中双侧接触要求 |
| 长时间悬停不敢下降 | y 速度长期为正或接近零，episode 超时 | 减轻 fuel_penalty 或 velocity_damping 对下降的限制，可加入轻柔的高度下降奖励 |
| 飞出视口 | x 坐标超出安全范围 | 增加对 x 偏移的急剧惩罚，甚至大常数负奖 |