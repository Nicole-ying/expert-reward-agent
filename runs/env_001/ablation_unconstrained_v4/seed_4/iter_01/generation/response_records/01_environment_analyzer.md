# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个2D平面内的着陆器轨迹优化任务。智能体初始位于视口顶部中央区域，带有随机初始力。核心目标是**到达并稳定地停泊在画面中央的目标垫板上**，同时尽可能减少引擎推力总消耗，并且在接触垫板时保持低速度和接近直立的姿态。次要目标包括：减少燃料使用、缩短完成任务时间，以及避免任何不安全接触。该任务**不是**持续的行走推进，也不是无明确到达目标的平衡存活，也不是多目标权重均等的复合任务；一切奖励设计最终服务于 “到达、减速、稳定着陆” 这一中心目的。

## 2. 任务类型选择
- selected_route_id: navigation_goal_reaching  
- confidence: high  
- reason: 任务的核心成功条件是到达并稳定在指定的目标垫板上，附属要求（节能、快速、姿态稳定）均为服务于一次成功到达的优化目标，不具有与到达目标并列的权重。符合导航目标到达类任务的特征，但不同于单纯导航，它强调到达后的“软接触”与稳定停泊。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（各维度均为浮点，接触标志以 0.0 或 1.0 的浮点形式出现）  
- 各维度含义与奖励可用性：  
  - obs[0]: x_position（相对目标垫板的水平坐标），reward_usable: true  
  - obs[1]: y_position（相对垫板高度的垂直坐标），reward_usable: true  
  - obs[2]: x_velocity（水平线速度），reward_usable: true  
  - obs[3]: y_velocity（垂直线速度），reward_usable: true  
  - obs[4]: body_angle（机体姿态角），reward_usable: true  
  - obs[5]: angular_velocity（角速度），reward_usable: true  
  - obs[6]: left_support_contact（左支撑点接触标志，1.0=已接触），reward_usable: true  
  - obs[7]: right_support_contact（右支撑点接触标志，1.0=已接触），reward_usable: true  

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 各动作含义（均为离散动作，不涉及连续幅度）：  
  - action 0: no_engine（不点火任何引擎）  
  - action 1: left_orientation_engine（点燃左侧定向引擎，主要用于顺时针或逆时针方向调整）  
  - action 2: main_engine（点燃主引擎，产生沿机体轴向的推力）  
  - action 3: right_orientation_engine（点燃右侧定向引擎，作用与左侧引擎相反）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  身体稳定停泊在目标垫板上（body_not_awake_or_settled 触发），且未发生 crash 或出界。  
- failure-like termination:  
  crash_or_body_contact（与地面或非目标区域的硬接触）、horizontal_position_outside_viewport（水平位置超出视口边界）。  
- ambiguous termination:  
  无特别模糊的情况，但**单独**依靠 body_not_awake_or_settled 无法区分成功与因卡死在错误位置而“静止”，需要结合位置和接触信号共同判断；另外在训练早期，可能因初始随机力漂移导致未能接触垫板而直接出界。  
- truncation:  
  未在描述中明确提及最大步数截断，但环境可能在超时后终止，具体截断逻辑被遮蔽，不应在奖励函数中隐式利用。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: info 为固定空字典 `{}`，**没有任何可用字段**。  
- forbidden_or_uncertain_info_fields: 官方 success/failure 标志、终止原因字符串等均不可用，因为步函数返回 info 为空。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs: 上一时刻的完整8维观察  
- action: 刚执行的动作（0~3的离散动作）  
- next_obs: 新时刻的完整8维观察  
- info: 只能安全地视为空字典，**不应从中读取任何值**  
- training_progress: 默认0.0，除非上层明确允许且需要，否则**避免使用**

禁止使用：
- original_reward: 该参数被遮蔽，不得作为信号原始值或基线  
- 任何 official_reward 的变体  
- 未在本卡片中声明的 info 字段（实际上全部禁止）  
- 任何假设的“成功”或“失败”标签  
- 对观察空间未定义的维度进行切片

## 7. 可用于奖励函数的信号
- position: 可通过 obs[0], obs[1] 及 next_obs[0], next_obs[1] 计算相对于目标的距离或位置变化。  
- velocity: 可通过 obs[2], obs[3] 与 next_obs[2], next_obs[3] 得到合速度大小或各分量。  
- orientation: obs[4], next_obs[4] 提供机体角度；obs[5], next_obs[5] 提供角速度。  
- contact: obs[6], obs[7] 与 next_obs[6], next_obs[7] 为两腿接触标志，可用于判定着陆状态。  
- action/engine: 动作序号可用于判断是否点燃主引擎（action==2）或定向引擎（action==1 或 3），从而计算燃料使用。  
- other: 由位置、速度、角度衍生出距离变化率（接近速度）、姿态变化等合成信号。

## 8. 不确定或不可用的信号
- 任何与“成功”或“失败”直接相关的标签（info 为空）。  
- 原环境奖励函数的具体数值或规律。  
- 环境内部物理参数（如质量、重力加速度、垫板大小）—— 这些属于先验知识且未被提供，只能从观察中隐式感知。  
- 时间步计数或总步数上限（除非上层在调用时作为额外参数显式传入，但不在标准接口内）。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D rigid body with thrusters
  actuator_type: one main bidirectional thruster + two orientation thrusters (left/right)
  contact_structure: two support legs with binary ground contact sensors
primary_objectives:
  - 使机身质心到达并保持在目标垫板水平范围内
  - 着陆时垂直接近速度足够低（避免硬碰撞）
  - 着陆后机身姿态保持接近竖直（角度、角速度小）
secondary_objectives:
  - 最小化引擎点火频率/总推力时间
  - 在保证安全前提下尽快完成着陆
main_failure_risks:
  - 直接坠毁在垫板以外的地形上
  - 超出水平视口边界而强制终止
  - 着陆姿态严重倾斜导致单腿悬空，虽“接触”但不稳定
  - 燃料不足导致无法减速而高速撞击（虽然在观察中不直接暴露燃料量，但从动作频率可部分推断）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: proximity_to_target  
  purpose: 鼓励质心不断靠近垫板中心（xy 距离减小），是达成“到达”目标的最直接推动力。  
  why_required: 没有位置激励，智能体将无法学会朝目标移动。  
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]]  
  risks: 仅用终点距离会使智能体不会减速，可能高速撞上垫板被判定 crash 或导致不稳定。

- role_id: soft_landing_velocity_penalty  
  purpose: 在接近地表（小 y 距离）且已至少有一只脚接触时，严惩高速度（尤其是垂直分量）。  
  why_required: 确保着陆满足“安全接触”的要求，是区分成功与失败的关键。  
  usable_signals: [obs[1], next_obs[1], obs[2:4], next_obs[2:4], contact signals]  
  risks: 如果全局启用可能阻碍早期快速下降，需要条件化启用。

- role_id: upright_stability_penalty  
  purpose: 惩罚过大的机体倾斜角度和角速度，尤其是在接触阶段。  
  why_required: 着陆器倾覆是典型失败模式，必须由奖励直接压制。  
  usable_signals: [obs[4], obs[5], next_obs[4], next_obs[5]]  
  risks: 过于敏感可能阻止必要的姿态调整。

### 10.2 条件职责 conditional_roles
- role_id: fuel_efficiency_penalty  
  purpose: 在不需要大幅调整时，惩罚不必要的引擎点火，以节省燃料。  
  condition_to_use: 当智能体已处于接近目标且速度足够低时，或不处在迫降阶段时启用。  
  usable_signals: [action, position, velocity signals]  
  risks: 过早使用会抑制学习探索动作导致无法到达目标。

- role_id: stable_stand_bonus  
  purpose: 当两腿均稳定接触且姿态竖直且速度接近零时，给予正向奖励推动“停稳”。  
  condition_to_use: 仅在明确检测到着陆成功倾向时启用，用于避免反复弹跳。  
  usable_signals: [next_obs[4:8], next_obs[2:4]]  
  risks: 如果触发条件过于宽松，可能奖励不完整着陆姿态。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: explicit_success_shaping  
  reason: 由于环境没有提供显式的 success 标志，无法从 info 中获得官方成功信号，强行通过硬编码边界（如“位置在垫板内且速度小且双触”即视为成功）会导致与真实物理终止不一致，甚至奖励假阳性。  
  forbidden_or_missing_signals: [explicit_success_flag]

- role_id: remaining_time_reward  
  reason: 未提供任务剩余步数或时间信息，环境也未声明时间奖励，避免引入不存在的进度赶时信号。  
  forbidden_or_missing_signals: [time_to_limit, internal_step_counter]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| proximity_to_target | next_obs[0:2] (x, y 距离)，obs[0:2] 用于增量 | – | distance_to_target (L2)， position_delta_reward (距离减小奖赏) | 注意 y 方向是相对垫板高度，y=0 代表机身已到达垫板高度线，此时距离仅剩水平偏移。 |
| soft_landing_velocity_penalty | next_obs[2:4] 的合速度/垂直分量，contact any (obs[6] 或 obs[7])，y_position | – | conditional_quadratic_penalty， multiplicative_activation (由接近地表和接触程度调节因子) | 必须仅在满足“接近地表且接触”条件时生效，防止过早减速。 |
| upright_stability_penalty | next_obs[4] (角度)，next_obs[5] (角速度) | – | absolute_value 或 quadratic_penalty | 可考虑在接触后进一步放大权重。 |
| fuel_efficiency_penalty | action == 1,2,3 (引擎点火) | 燃料
