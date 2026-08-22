# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 离散控制轨迹优化问题。主体从视口顶部中央附近随机初始速度开始，目标是尽快到达中央的目标着陆垫并稳定停靠，同时尽量减少发动机推力使用。智能体需要学会接近目标、降低速度、维持稳定姿态并实现安全接触。  
**主目标**：以稳定姿态降落在目标区域，实现安全着陆。  
**次要目标**：最小化发动机推力（燃料消耗）。  
**不应混淆**：单纯的速度最小化或姿态抖动惩罚不是独立目标，而是安全着陆的组成部分；燃料节省是次要，但必须在成功着陆前提下追求。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是朝向固定目标位置（着陆垫）移动并停靠在上面，附属有速度、姿态、燃料约束，但没有其他权重相当的冲突目标。到达目标是主线，其他要求是为了保证到达质量，因此属于导航目标到达类。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32 (根据常见环境约定，每个维度为浮点数，接触标志也是浮点 0.0/1.0)
- obs[0]: x_position，主体中心相对于目标着陆垫中心的水平坐标，reward_usable: true
- obs[1]: y_position，主体中心相对于着陆垫高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity，水平线速度，reward_usable: true
- obs[3]: y_velocity，垂直线速度，reward_usable: true
- obs[4]: body_angle，主体朝向角（弧度），reward_usable: true
- obs[5]: angular_velocity，角速度，reward_usable: true
- obs[6]: left_support_contact，左支撑腿接触标志（1.0 接触，0.0 不接触），reward_usable: true
- obs[7]: right_support_contact，右支撑腿接触标志，reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不启动任何发动机
- action 1: left_orientation_engine — 启动左侧姿态发动机（产生逆时针力矩）
- action 2: main_engine — 启动主发动机（产生向上的推力，同时可能影响姿态）
- action 3: right_orientation_engine — 启动右侧姿态发动机（产生顺时针力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，但 `body_not_awake_or_settled` 条件下若双腿均接触且速度足够低则很可能表示成功着陆；该条件由引擎自动触发，与 `crash_or_body_contact` 互斥。
- failure-like termination: `crash_or_body_contact`（主体与地面或非着陆垫区域碰撞）、`horizontal_position_outside_viewport`（水平飞出边界）均明确为失败。
- ambiguous termination: `body_not_awake_or_settled` 本身只表示主体停止运动或稳定，需结合腿部接触状态和速度判断是否成功。
- truncation: 无时间截断（未提及 episode length 限制）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 字典为空，无任何标志）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 固定为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 全部；只能通过观测序列推断成功/失败。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs, next_obs：当前与下一时刻观测向量
- action：当前动作（0-3）
- info：仅有 `{}`，可忽略
- training_progress：若 prompt 明确允许则可用，本环境未要求，默认禁止

禁止使用：
- original_reward（官方奖励已屏蔽，严禁作为信号）
- 任何未在 observation_space 中声明的字段
- 任何非空的 info 字段（info 为空）

## 7. 可用于奖励函数的信号
所有观测维度均可用于构造奖励函数，具体分块如下：
- position: obs[0] (x_position), obs[1] (y_position) — 用于距离目标度量
- velocity: obs[2] (x_velocity), obs[3] (y_velocity) — 用于速度惩罚
- orientation: obs[4] (body_angle) — 姿态偏离惩罚
- angular velocity: obs[5] — 角速度稳定惩罚
- contact: obs[6], obs[7] — 可用于成功检测与接地奖励
- action/engine: action 本身可映射为燃料消耗惩罚

## 8. 不确定或不可用的信号
- 不确定：通过 `body_not_awake_or_settled` 精确判断成功的规则（需结合多维度阈值，无绝对正确标准），但环境本身不提供 success 标签，需由奖励设计者约定。
- 不可用：original_reward、任何 info 字段。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 刚体主体，带两条对称支撑腿
  actuator_type: 一个主发动机（向上推力）、两个姿态发动机（对称力矩）
  contact_structure: 两条腿与地面接触点，可分别检测接触
primary_objectives:
  - 水平与垂直位置接近目标着陆垫中心
  - 着陆时垂直接近零平移速度，水平速度低，角速度接近零
  - 双腿同时接触着陆垫
secondary_objectives:
  - 最小化总发动机推力使用（省燃料）
main_failure_risks:
  - 碰撞地面或着陆区以外
  - 飞出水平边界
  - 在未接触情况下过早停止（陷入局部最优）
  - 时间过长消耗过多燃料
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: goal_distance_reduction  
  purpose: 鼓励主体向目标着陆垫移动，缩小位置误差  
  why_required: 任务是目标到达，位置偏差是主要驱动信号  
  usable_signals: [x_position, y_position]  
  risks: 单独使用可能产生过冲或剧烈摆动不减速

- role_id: velocity_attenuation_near_goal  
  purpose: 在接近目标时降低速度，避免高速撞击  
  why_required: 安全着陆需要低速接触，否则视为失败  
  usable_signals: [x_velocity, y_velocity, x_position, y_position]  
  risks: 过早惩罚速度可能抑制必要的机动性；需与距离耦合设计

- role_id: orientation_stabilization  
  purpose: 保持主体接近直立，防止倾倒  
  why_required: 双腿着陆需要主体角度接近零，且角速度收敛  
  usable_signals: [body_angle, angular_velocity]  
  risks: 对姿态要求过严可能阻碍必要的调整动作

- role_id: thrust_efficiency  
  purpose: 惩罚发动机使用，鼓励节省燃料  
  why_required: 任务明确要求尽可能少用推力，但重要性次于着陆本身  
  usable_signals: [action]  
  risks: 惩罚过重可能导致智能体选择“不动作等死”而不是尽快着陆，需权衡权重

### 10.2 条件职责 conditional_roles
- role_id: terminal_success_bonus  
  condition_to_use: 仅在 episode 终止且通过观测判断为成功着陆时（双腿接触、速度极低、姿态近零）给予一次性正向奖励  
  usable_signals: [left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle, done (隐含)]  
  risks: 伪成功条件设置不严格可能导致奖励发放给局部不动状态；没有 done 传入 reward 函数时需通过迭代外部实现，增加复杂度

### 10.3 慎用/禁用职责 avoid_roles
- role_id: use_official_reward  
  reason: 官方奖励已强制屏蔽，不可访问，任何尝试使用 original_reward 均违反规则  
  forbidden_or_missing_signals: [original_reward]

- role_id: info_based_success_detection  
  reason: info 字段为空，不存在任何成功或失败标志，无法依赖  
  forbidden_or_missing_signals: [info.success, info.failure]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| goal_distance_reduction | x_position, y_position | — | dense_state_signal, bounded_signal (e.g., -norm) | 可结合距离绝对值或平方 |
| velocity_attenuation_near_goal | x_velocity, y_velocity, x_position, y_position | — | squared_penalty, distance-gated penalty | 距离阈值可调 |
| orientation_stabilization | body_angle, angular_velocity | — | quadratic_penalty | 可叠加 |
| thrust_efficiency | action | — | discrete_penalty (per action) | 主发动机惩罚应高于姿态发动机 |
| terminal_success_bonus | left_support_contact, right_support_contact, x_velocity, y_velocity, body_angle | done flag in reward function | sparse_event_bonus | 需外部逻辑判断成功并传入奖励中，或作为最终状态奖励一次性发放 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 主体悬停在目标上方不降落 | y_position 保持较高，速度接近零，双腿接触为 0 | 降低高度惩罚权重，或引入“下降不积极”的小惩罚，或增加成功后稀疏奖励 |
| 主体猛烈撞击着陆垫 | 终止时 y_velocity 很大或 legs 接触瞬间速度陡增 | 加大速度惩罚的比例，或者在接触瞬间要求速度必须低于阈值 |
| 仅单腿着陆后倾倒 | 终止时一条腿接触，另一条无，body_angle 大 | 加入双腿接触的条件惩罚，或提高姿态保持权重 |
| 主体长时间振荡无法稳定 | x_position 和 x_velocity 高频变化，角速度振荡 | 提高角速度阻尼惩罚，或对动作变化率加以约束 |
| 总是选择“不动作”存活但进度缓慢 | 大量步数无明显位置变化，燃料消耗惩罚过重 | 调低燃料惩罚，或增加基于进度的潜在势能奖励 |
| 飞出水平边界 | 终止条件触发，x_position 超出阈值 | 强化位置保持惩罚，或在边界内增加约束力信号 |
