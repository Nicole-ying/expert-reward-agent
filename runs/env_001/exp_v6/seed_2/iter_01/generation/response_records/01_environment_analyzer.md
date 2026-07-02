# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视口顶部中心区域出发，尽快抵达并稳定停在中央目标垫上，同时尽量减少引擎推力。  
理想行为：快速靠近目标 → 减速 → 保持竖直姿态 → 左右支撑腿同时接触目标垫 → 稳定停靠。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是导航到固定目标点（中央目标垫）并稳定停靠，属于目标到达型任务；燃料效率是辅助目标，但不改变主类型。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- 维度含义（按索引）：
  - obs[0] (x_position)：目标垫水平坐标的相对值（单位为环境内部尺度）
  - obs[1] (y_position)：相对于目标垫高度的高度坐标
  - obs[2] (x_velocity)：水平方向线速度
  - obs[3] (y_velocity)：竖直方向线速度
  - obs[4] (body_angle)：机体倾角（弧度或角度，环境未明确单位，通常可能为弧度）
  - obs[5] (angular_velocity)：角速度
  - obs[6] (left_support_contact)：左支撑腿接触标志，1.0 表示接触，0.0 表示未接触
  - obs[7] (right_support_contact)：右支撑腿接触标志，1.0 表示接触，0.0 表示未接触

## 4. 动作空间 action_space
- type: Discrete(4)
- 动作含义：
  - action 0：不点火（no_engine），仅依赖惯性和重力
  - action 1：左姿态引擎（left_orientation_engine），产生旋转力矩，用于调整机体倾角
  - action 2：主引擎（main_engine），产生主要推力，影响水平和竖直速度
  - action 3：右姿态引擎（right_orientation_engine），与左姿态引擎相反方向的旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination：  
  可能由 `body_not_awake_or_settled` 触发，若此时机体已稳定在目标垫（两侧接触、低速度、小倾角），可视为成功停靠。
- failure-like termination：  
  - `crash_or_body_contact`：发生碰撞或不当身体接触（非目标垫接触）  
  - `horizontal_position_outside_viewport`：水平坐标超出视口范围（飞出边界）
- ambiguous termination：  
  若 `body_not_awake_or_settled` 在非目标位置触发，可能是失败，但环境未提供显式区分信号。
- truncation：环境没有提供 truncation 信息，在 step 返回中第四个元素为 False，表明不存在时间截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  （info 为空字典，没有 success 字段）
- explicit_failure_flag_available: false  （同样没有 failure 标记）
- allowed_info_fields: {} （当前 info 不提供任何额外信息）
- forbidden_or_uncertain_info_fields:  
  - 所有 `info` 字段均因信息缺失而不可用，严禁在奖励函数中依赖 `info` 内容。  
  - 终止原因（crash、out of bounds、settled）不能作为奖励信号，因为这些条件不出现在 `obs` 或 `info` 中，且环境未提供终止理由的回调。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`：当前时刻的 8 维观察
- `action`：当前执行的动作 ID（0-3）
- `next_obs`：下一时刻的 8 维观察
- `info` 中明确允许的字段（当前为空，故不允许使用任何 info 字段）
- `training_progress`：除非本环境 prompt 明确允许，否则禁止用于奖励（当前未允许）

禁止使用：
- `original_reward`（被屏蔽的官方奖励）
- 任何未声明的 `info` 字段
- 任何未在 observation_space 中声明的观察切片
- 显式终止标志（因为不可得）

## 7. 可用于奖励函数的信号
- **位置**：`next_obs[0]`、`next_obs[1]`（相对目标垫的距离）
- **速度**：`next_obs[2]`、`next_obs[3]`
- **姿态**：`next_obs[4]`（角度）、`next_obs[5]`（角速度）
- **接触**：`next_obs[6]`、`next_obs[7]`（左右支撑接触标志，可判断是否平稳着陆）
- **动作/引擎**：`action`（用于惩罚推力使用）
- **变化量**：`obs` 与 `next_obs` 之间的差值（速度变化、位置变化），可间接反映控制效果

## 8. 不确定或不可用的信号
- **显式成功/失败标记**：`info` 中不存在 `success`、`failure` 或 `termination_reason` 字段
- **终止原因**：`crash_or_body_contact`、`horizontal_position_outside_viewport`、`body_not_awake_or_settled` 的具体类型在观测中不可直接获取
- **环境边界参数**：视口范围未知，无法仅凭 `obs` 准确判断是否即将超出边界（但可通过位置量级和安全阈值做近似）
- **目标垫的实际尺寸**：仅知相对坐标，无法获取垫子大小，因此接触标志的物理意义需通过实验确认
- **物理参数**：重力、推力大小、质量、惯性等均不提供，无法用于精确建模
