# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本环境是一个 **2D 类飞行器轨迹优化任务**。  
主体从一个靠近顶部中央的随机位置出发，带有初始随机作用力。  
核心目标是 **到达中央的目标着陆垫并保持稳定停靠（settle）**，同时尽可能 **快速** 且 **消耗最少的引擎推力**。  
智能体需要学会接近目标、减小速度、保持竖直姿态并安全接触。  
混淆目标：不要将“纯粹节约燃料”或“纯粹高速飞行”当作唯一目标，必须在安全着陆的前提下平衡时间与能耗。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务的核心是到达指定目标位置并停留，属于典型的导航到达类；虽然存在能耗和速度的次要要求，但其权重明确低于成功到达与安全停靠，不构成多目标任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，未显式说明但符合常规）
- obs[0]: x_position — 相对目标垫中心的水平坐标，可用于奖励函数 (reward_usable: true)
- obs[1]: y_position — 相对目标垫高度的垂直坐标，可用于奖励函数 (reward_usable: true)
- obs[2]: x_velocity — 水平速度，可用于奖励函数 (reward_usable: true)
- obs[3]: y_velocity — 垂直速度，可用于奖励函数 (reward_usable: true)
- obs[4]: body_angle — 机体朝向角，可用于奖励函数 (reward_usable: true)
- obs[5]: angular_velocity — 角速度，可用于奖励函数 (reward_usable: true)
- obs[6]: left_support_contact — 左侧支撑腿接触标志（0或1），可用于奖励函数 (reward_usable: true)
- obs[7]: right_support_contact — 右侧支撑腿接触标志（0或1），可用于奖励函数 (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不施加推力
- action 1: left_orientation_engine — 启动左方向引擎，通常产生旋转力矩（可能伴随微小侧向力）
- action 2: main_engine — 启动主引擎，产生向上的推力
- action 3: right_orientation_engine — 启动右方向引擎，通常产生反向旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` — 机体稳定不活动且可能已停靠在垫子上。该条件本身不一定表明成功，但若同时位置靠近目标垫，则极大概率是成功着陆。
- failure-like termination:  
  `crash_or_body_contact` — 机体发生碰撞或非预期身体接触（如侧面触地），可能是坠毁。  
  `horizontal_position_outside_viewport` — 水平坐标超出视口边界，视为飞出有效区域，失败。
- ambiguous termination:  
  上述条件被合并为一个布尔标志，无法在 `info` 中直接区分终止原因。
- truncation: 无显式截断（step 返回 `False` for truncation），只有终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回空字典 `{}`），因此禁止使用任何 info 字段。
- forbidden_or_uncertain_info_fields: 所有 info 字段（`success`, `failure`, `termination_reason` 等）均不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- `obs`：当前观测（8 维向量）
- `action`：当前执行的动作（离散 0-3）
- `next_obs`：下一时刻观测（可选用于差分或接触变化检测）
- `training_progress`：仅当 prompt 明确允许时可用于调度奖励强度（本环境未要求使用）

**禁止使用：**
- `original_reward`（`masked_reward`）—— 已被遮蔽，不可复制或参考
- `info` 中的任何字段（因为 info 为空）
- 任何未在观测空间中声明的数据或符号

## 7. 可用于奖励函数的信号
- **position**：`obs[0]`(x相对位置)，`obs[1]`(y相对位置) —— 可构建到目标垫的距离、高度偏差。
- **velocity**：`obs[2]`(x速度)，`obs[3]`(y速度) —— 可构建总速度或速度分量，用于鼓励低速着陆。
- **orientation**：`obs[4]`(角度)，`obs[5]`(角速度) —— 可促使保持竖直。
- **contact**：`obs[6]`(左接触)，`obs[7]`(右接触) —— 可评估是否已着陆或着陆质量。
- **action/engine**：动作本身以及 `action != 0` 的布尔值 —— 可直接用于惩罚引擎使用。
- **other**：可利用 `next_obs` 与 `obs` 的差分（如位置变化、接触变化）来检测状态转移，但不推荐依赖复杂构造。

## 8. 不确定或不可用的信号
- **明确的 crash 标记**：虽然存在 `crash_or_body_contact` 终止逻辑，但该信息不进入观测或 info，无法可靠判断机身是否因碰撞而终止。
- **距离/燃料显式读数**：没有直接的燃料剩余或推进器功率信息。
- **环境内置 success**：info 为空，无 `success` 标志，无法直接奖励成功事件。
- **目标垫绝对坐标**：仅提供相对位置，无法直接获取全局边界。
- **推力大小**：动作仅为离散引擎开闭，无法获取推力物理量。

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: 2D autonomous vehicle/lander with two support legs
  actuator_type: discrete impulse engines (main upward thrust, left/right orientation)
  contact_structure: two-point support contact (left & right legs), prone to tipping
primary_objectives:
  - reach the target pad (minimize final x,y offset)
  - settle at low speed with upright orientation
secondaary_objectives:
  - minimize time-to-land (encourage fast approach)
  - minimize fuel/thrust usage (encourage efficient actions)
main_failure_risks:
  - high-velocity impact causing crash
  - landing far from target pad (horizontal drift)
  - tipping over due to excessive angular velocity or steep angle
  - flying out of the horizontal boundary
```

## 10. 奖励职责拆
