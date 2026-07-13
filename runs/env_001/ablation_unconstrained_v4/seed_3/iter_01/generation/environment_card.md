# 匿名环境理解卡片

## 1. 任务目标
本任务是一个二维飞行器的轨迹优化问题。主体从视口上方中央附近以随机初始受力出发，需要在最短时间内到达并稳定在中央目标着陆垫上，同时尽量少用引擎推力。智能体必须学会向目标靠近，减小水平和垂直速度，保持稳定姿态，并实现安全接触（双支撑腿同时触垫）。次要目标是节约燃料（即减少引擎使用次数），但绝不能以牺牲成功着陆和安全为代价。

## 2. 任务类型选择
selected_route_id: `navigation_goal_reaching`  
confidence: high  
reason: 核心目标是到达并稳定在指定目标垫，属于典型的导航/目标到达任务；燃料效率和时间因素为辅助优化，不构成权重相当的冲突多目标。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: 默认 float32（连续值，接触标志为 0.0 或 1.0）  
- 各维度含义：
  - obs[0]：`x_position`，相对目标垫的水平坐标（负/正代表左右），可用于奖励计算 `reward_usable: true`
  - obs[1]：`y_position`，相对垫高度的垂直坐标，可用于奖励计算 `reward_usable: true`
  - obs[2]：`x_velocity`，水平线速度，可用于奖励计算 `reward_usable: true`
  - obs[3]：`y_velocity`，垂直线速度，可用于奖励计算 `reward_usable: true`
  - obs[4]：`body_angle`，机体偏转角（以弧度计），可用于奖励计算 `reward_usable: true`
  - obs[5]：`angular_velocity`，角速度，可用于奖励计算 `reward_usable: true`
  - obs[6]：`left_support_contact`，左支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`
  - obs[7]：`right_support_contact`，右支撑腿接触标志（0.0/1.0），可用于奖励计算 `reward_usable: true`

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- 动作含义：
  - 0：`no_engine`，不做任何推力  
  - 1：`left_orientation_engine`，点燃左姿态引擎（通常产生顺时针力矩和微小侧向力）  
  - 2：`main_engine`，点燃主引擎（通常产生向上的主要推力）  
  - 3：`right_orientation_engine`，点燃右姿态引擎（通常产生逆时针力矩和微小侧向力）  

说明：动作是离散的推进器选择，每个时间步只能选择一种操作。因此奖励中无法直接获得连续推力值，但可基于动作类型判断是否使用了引擎，从而惩罚燃料消耗。

## 5. step 与终止条件分析
### 5.1 终止模式
根据 `termination_conditions` 原文，共三种终止原因：
1. **crash_or_body_contact（碰撞或身体接触）**  
   - 很可能是与地面或障碍物发生不当碰撞，通常代表失败。
2. **horizontal_position_outside_viewport（水平位置超出视口）**  
   - 机体横向飞出边界，代表失败。
3. **body_not_awake_or_settled（机体未醒或已稳定）**  
   - 语义存疑：可能表示机体已进入休眠状态（着陆后稳定静止）或未能稳定（翻倒后停止）。由于原文将它与前两类并列且未注明成功/失败，该终止条件本身是 **模糊的**。在典型同类任务中，机体成功着陆后会进入“睡眠”（awake=False），因而该条件可能是 **成功终止**。但在正式设计 reward 时，不能直接依赖该条件为成功标志，必须结合接触、位置、速度等观测自行判断。

### 5.2 步骤返回信息
`step()` 返回 `(state, masked_reward, terminated, False, {})`，其中 `info` 为空字典。因此：
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无
- **forbidden_or_uncertain_info_fields**: 所有 info 字段（因 info 为空，无可用信号）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

- 允许使用：
  - `obs`：当前观测（8维向量）
  - `action`：所选离散动作（0~3 整数）
  - `next_obs`：下一时刻观测
  - `info`：空字典，因此无可用字段
  - `training_progress`：仅当 prompt 明确允许时使用（本例未允许，应采用 0.0 或忽略）
- 禁止使用：
  - `original_reward`（官方奖励被掩码，不可使用、不可还原）
  - 任何未声明的 info 字段
  - 任何未声明的观测切片或全局变量

## 7. 可用于奖励函数的信号
基于 `obs`、`action` 和 `next_obs`，可提取以下信号：
- **位置信号**：`obs[0]`、`obs[1]`（或 `next_obs[0]`、`next_obs[1]`），表示相对目标垫的水平和垂直偏差。
- **速度信号**：`obs[2]`、`obs[3]`，水平与垂直线速度。
- **姿态信号**：`obs[4]`（偏转角）、`obs[5]`（角速度）。
- **接触信号**：`obs[6]`、`obs[7]`，左右支撑腿是否触地。
- **动作/引擎信号**：`action` 值，用于判断是否使用了引擎（0为无推力，1、2、3均消耗燃料）。
- **附加信号**：还可以计算 `next_obs` 与目标的距离变化、速度变化等差分信号。

所有这些信号均源自观测，可直接用于塑造奖励，无需依赖 info。

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不存在于 info 或 step 返回中，不能直接使用。
- **剩余燃料值**：环境中未提供，因此无法直接基于燃料余量设计奖励。
- **目标垫绝对坐标**：任务描述已暗示目标垫在画面中央，但观测是相对值，绝对坐标不可用（不影响任务）。
- 因 info 为空，任何类似 `info["success"]` 或 `info["termination_reason"]` 的信号均不可用。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: lander_with_two_legs
  actuator_type: main_thruster_and_orientation_thrusters
  contact_structure: two_leg_contacts
primary_objectives:
  - 到达目标垫中心：x → 0, y → 0
  - 稳定着陆：双支撑腿触地，小速度，小倾斜角
secondary_objectives:
  - 节省燃料：最小化非零动作的使用次数
  - 快速到达：隐含在任务描述中，但不可牺牲安全性
main_failure_risks:
  - 与地面或障碍物猛烈碰撞（crash_or_body_contact）
  - 横向飞出可用区域（horizontal_position_outside_viewport）
  - 着陆后翻倒或未能垂直稳定，导致 body_not_awake_or_settled 但状态不合格
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- **role_id: `goal_approach`**  
  purpose: 引导 agent 向目标垫中心靠近（x→0, y→0）  
  why_required: 是任务核心，没有它 agent 无法学会到达目标  
  usable_signals: [obs[0], obs[1], next_obs[0], next_obs[1]]  
  risks: 单独使用可能导致高速冲撞目标，需配合减速职责

- **role_id: `soft_landing`**  
  purpose: 在接近目标时鼓励低速、小角度、双脚触地  
  why_required: 确保安全着陆，避免碰撞或弹开  
  usable_signals: [obs[2], obs[3], obs[4], obs[6], obs[7], next_obs[2], next_obs[3], next_obs[4], next_obs[6], next_obs[7]]  
  risks: 若在全局范围惩罚速度，可能抑制探索；应在接近目标区时增强权重

- **role_id: `successful_termination_bonus`**  
  purpose: 在满足成功着陆条件时给予巨大正向奖励  
 