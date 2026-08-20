# Env_001 环境理解卡片

## 1. 任务目标
控制一个2D飞行器（或着陆器），从屏幕顶部中央附近出发（初始带有随机推力），学会**尽可能快地**、**用尽可能少的引擎推力**到达并稳定降落在中心的目标平台上。期望的行为是：接近目标、减速、保持机体竖直、两条腿安全接触平台并静止。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务目标明确包含了三个需要平衡的子目标——快速到达（时间效率）、最小化引擎使用（能量效率）以及安全稳定接触（着陆安全性），属于典型的多目标优化任务。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float32（其中接触标志为 0.0/1.0，但以浮点数存储）  
- obs[0]: **x_position** – 机体相对于目标平台的水平坐标（向右为正）  
- obs[1]: **y_position** – 机体相对于目标平台高度的垂直坐标（向上为正）  
- obs[2]: **x_velocity** – 水平线速度  
- obs[3]: **y_velocity** – 垂直线速度  
- obs[4]: **body_angle** – 机体朝向角度（弧度制，0 表示竖直向上）  
- obs[5]: **angular_velocity** – 角速度  
- obs[6]: **left_support_contact** – 左侧支撑腿接触标志（1.0 表示接触）  
- obs[7]: **right_support_contact** – 右侧支撑腿接触标志（1.0 表示接触）

## 4. 动作空间 action_space
- type: Discrete(4)  
- action 0: **no_engine** – 不启动任何引擎，让机体依靠惯性和重力自由运动  
- action 1: **left_orientation_engine** – 启动左侧姿态引擎（通常产生顺时针/逆时针旋转力矩）  
- action 2: **main_engine** – 启动主引擎（通常提供向上的推力）  
- action 3: **right_orientation_engine** – 启动右侧姿态引擎（产生与 left 相反方向的旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: `body_not_awake_or_settled` – 机体休眠或稳定下来，通常意味着速度与角速度都非常低，可能已经成功着陆并静止；但必须结合位置、角度、接触信号来确认是否真正降落在目标平台上。
- **failure-like termination**:  
  - `crash_or_body_contact` – 可能指机体与地面或其他障碍物发生了不安全接触（例如高速撞击、单腿触地、侧翻等），通常是失败。  
  - `horizontal_position_outside_viewport` – 飞出可视区域边界，明确为失败。
- **ambiguous termination**: `crash_or_body_contact` 本身含义模糊，因为安全着陆也会产生接触；需要根据下一时刻观测的速度、角度、双腿接触情况区分为“安全成功接触”或“危险失败碰撞”。
- **truncation**: 在提供的 step 源码中 `truncated` 恒为 `False`，无时间步数截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空字典，无任何显式成功标记）  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 不可用）  
- forbidden_or_uncertain_info_fields: 所有 info 键值均不可用，禁止假设存在 `success`、`failure`、`termination_reason` 等字段。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用：**
- `obs`：当前时刻的 8 维观测数组（含义见第 3 节）  
- `action`：该步执行的动作（0~3）  
- `next_obs`：下一时刻的 8 维观测数组  
- `info`：但当前版本中 info 为空字典，实际不可从中读取任何字段  
- `training_progress`：**仅在任务描述明确提示允许按训练进度调整奖励时可用**，当前任务描述未提及，故应禁止使用  

**禁止使用：**
- `original_reward`（即官方原始奖励，已被屏蔽，不可访问）  
- 任何未在第 3 节中声明的 obs 维度切片（例如不要假设 obs[8] 等）  
- 任何 info 中的键值（包括 `success`、`termination_reason` 等）

## 7. 可用于奖励函数的信号
基于 `obs`、`action` 和 `next_obs`，以下信号是清晰可用且与任务目标相关的：
- **位置信号**：`next_obs[0]`（水平误差）、`next_obs[1]`（垂直误差），可计算到目标平台的欧氏距离、水平/垂直偏差等。
- **速度信号**：`next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度），可用于鼓励末端低速、惩罚高速撞击。
- **姿态信号**：`next_obs[4]`（机体角度）、`next_obs[5]`（角速度），鼓励角度趋近于 0（竖直）、角速度小（稳定）。
- **接触信号**：`next_obs[6]`（左腿接触）、`next_obs[7]`（右腿接触），可奖励双腿同时接触（安全着陆），惩罚单腿或不接触。
- **动作/引擎信号**：根据 `action` 是否为 1、2、3 可获知是否使用了引擎，可用于惩罚不必要的推力（能量消耗），鼓励 `action=0` 当已经安全着陆时。
- **复合安全指标**：结合位置、速度、角度和接触，可以设计“成功着陆”的近似判断（如距离 < 阈值、速度 < 阈值、|角度| < 阈值、双腿接触均为 1.0）。

## 8. 不确定或不可用的信号
- 原始环境的内置奖励函数被完全屏蔽，其具体数值逻辑不可知，禁止尝试还原或模仿。
- info 字典为空，无法提供任何关于 episode 成功、失败、终止原因的额外信息。
- 没有显式的“着陆成功/失败”标签，只能通过观测组合进行启发式推断。