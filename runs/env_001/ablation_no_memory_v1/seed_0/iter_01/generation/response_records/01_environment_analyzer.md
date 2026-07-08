# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个 2D 飞行器从视口顶部中心区域出发（初始带有一定随机速度），尽快抵达中央的着陆垫并稳定着陆。  
要求：以尽可能快的速度接近目标，减少移动速度，保持稳定姿态，最后安全地接触到着陆垫。  
此外，在整个过程中尽量少使用主引擎推力，实现节能。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box（连续值）
- shape: (8,)
- dtype: float32
- 每一维含义：
  - obs[0]: x_position — 飞行器相对于着陆垫中心的水平坐标（负左正右）
  - obs[1]: y_position — 飞行器相对于着陆垫表面高度的垂直坐标（正值在上方）
  - obs[2]: x_velocity — 水平线速度
  - obs[3]: y_velocity — 竖直线速度
  - obs[4]: body_angle — 机体倾斜角度
  - obs[5]: angular_velocity — 机体角速度
  - obs[6]: left_support_contact — 左支撑/接触标志（1.0 接触，0.0 不接触）
  - obs[7]: right_support_contact — 右支撑/接触标志（1.0 接触，0.0 不接触）

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- 动作含义：
  - action 0: no_engine — 不启动任何引擎，只受重力/惯性影响
  - action 1: left_orientation_engine — 启动一个姿态引擎，使机体逆时针（或某一方向）旋转
  - action 2: main_engine — 启动主引擎，产生与机体朝向相关的推力（通常向上）
  - action 3: right_orientation_engine — 启动另一个姿态引擎，使机体相反方向旋转

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  没有显式的成功标记。但在任务描述中，“到达着陆垫并稳定接触”是目标。  
  当机体在垫上方、两侧接触标志均为1、且速度/角速度极小、位置稳定（即 `body_not_awake_or_settled`）时，环境会因 `body_not_awake_or_settled` 而终止。这种情况很可能对应成功着陆（需结合实际行为判断）。
- failure-like termination:
  - `crash_or_body_contact`：可能指机体与除着陆垫以外的地面或墙壁发生碰撞（如侧翻、头部碰撞等），这应属于失败。
  - `horizontal_position_outside_viewport`：飞行器水平位置超出视野边界，显然失败。
- ambiguous termination:  
  - `body_not_awake_or_settled` 这个终止条件本身不区分是成功着陆还是单纯因机体“睡着”（如动作太弱、陷入死区）。如果出现在没有着陆接触、远离目标点时，可能不代表成功；若出现在垫上且接触时，代表成功。因此单独看终止信号不能直接作为奖励依据。
- truncation: 未提供（本次环境没有设定最大步数截断，或已融入终止条件中）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `{}` （step 返回的 info 为空字典，可用字段无）
- forbidden_or_uncertain_info_fields: 任何自定义字段（如 `success`、`landed`、`crash` 等）均不可用，因为环境未提供

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`：当前步的 8 维观察（每维含义见上方）
- `action`：当前步采取的动作编号（0~3）
- `next_obs`：下一步的 8 维观察
- `info`：空字典，无可依赖字段
- `training_progress`：若 prompt 明确允许，否则禁止使用

禁止使用：
- `original_reward`（官方奖励被屏蔽，不可访问、不可重建）
- 任何未在 `observation_space` 中声明的变量
- 任何未在 `info` 中出现的字段
- 对 obs 中未明确声明的维度切片

## 7. 可用于奖励函数的信号
- **位置（相对目标）**：`next_obs[0]` (x_position) 和 `next_obs[1]` (y_position) 可用于评判距着陆垫中心的距离
- **速度**：`next_obs[2]` (x_velocity) 和 `next_obs[3]` (y_velocity) 可用于评判机体在着陆时的平稳性
- **姿态**：`next_obs[4]` (body_angle) 和 `next_obs[5]` (angular_velocity) 可用于评判姿态是否保持水平/稳定
- **接触**：`next_obs[6]` (left_support_contact) 和 `next_obs[7]` (right_support_contact) 可用于检测是否着落到垫上（双接触通常说明成功着陆）
- **动作/引擎**：`action` 可用于衡量引擎使用情况，特别是 action 2（主引擎）的使用频率

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：info 中无相关字段，只能通过终止条件组合和接触、位置状态来间接推断
- **绝对世界坐标**：只提供了相对于目标的坐标，无法知道全局顶点位置（但不影响任务）
- **真实环境奖励**：被屏蔽，无法使用
- **着陆垫的绝对坐标**：未给出，仅能从 x, y 相对值推理
- **剩余步数/时间信号**：没有提供，不能作为 progress 相关信号（除非 prompt 允许 training_progress）
