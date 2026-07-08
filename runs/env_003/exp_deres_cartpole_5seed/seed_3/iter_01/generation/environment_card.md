# Env_003 环境理解卡片

## 1. 任务目标
在一个水平的直线轨道上，通过向移动底座施加固定大小的水平力（离散两个方向），控制底座移动，使底座顶端通过无动力关节连接的摆杆尽可能长时间地保持直立（即杆偏离竖直的角度不超过阈值），同时底座本身不能超出轨道边界。**本质是一个生存平衡任务：最大化存活步数（或时间）。**

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求持续维持系统在不稳定平衡点附近，没有预设目标位置或导航点，终止仅由平衡状态破坏或轨道越界触发，奖励稀疏且与存活时长正相关，属于典型的生存/平衡任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (4,)
- dtype: float32
- 各维度含义（按索引）：
  - obs[0]: 底座水平位置 (base_position)，取值范围约 [-4.8, 4.8]，但有效安全范围实际为 [-2.4, 2.4]，超出则终止。
  - obs[1]: 底座水平速度 (base_velocity)，无界。
  - obs[2]: 杆相对于竖直方向的偏角 (pole_angle)，单位弧度，范围约 [-0.4189, 0.4189]；终止阈值为 [-0.20944, 0.20944]。
  - obs[3]: 杆角速度 (pole_angular_velocity)，无界。

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 2
- 动作含义：
  - action 0: 向轨道负方向施加固定大小的水平力 (push_negative_direction)
  - action 1: 向轨道正方向施加固定大小的水平力 (push_positive_direction)

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: **无**。环境不定义“成功”概念，存活得越久越好。
- failure-like termination:
  - 底座位置绝对值超过 2.4（即底座掉出轨道有效区域）。
  - 杆偏转角绝对值超过 0.20943951 弧度（约 12°）。
  - 以上任一发生均视为**平衡失败**，episode 终止。
- ambiguous termination: 无。
- truncation: episode 达到 500 步后强制截断，属于时间截断，不代表任务失败。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: info 为空字典 `{}`，无任何额外字段。
- forbidden_or_uncertain_info_fields: 所有在 info 中未声明的字段（例如 success、failure、termination_reason、reward 等）均不存在，禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前步观察
- `action`：当前步执行的动作
- `next_obs`：下一步观察
- `info`：空字典，但接口允许传入（实际无有效字段）
- `training_progress`：**默认禁止**，除非任务 prompt 明确允许。本项目 prompt 未提及可以使用，故禁止使用。

禁止使用：
- `original_reward`：官方奖励被隐藏，不可用。
- 任何未在 info 或 obs 中声明的字段。
- 任何未在观察空间中明确列出的 obs 切片。

## 7. 可用于奖励函数的信号
- **底座位置**：`obs[0]` / `next_obs[0]`，可用于惩罚远离中心。
- **底座速度**：`obs[1]` / `next_obs[1]`，可用于平滑控制。
- **杆偏角**：`obs[2]` / `next_obs[2]`，核心平衡信号。
- **杆角速度**：`obs[3]` / `next_obs[3]`，可用于倾向稳定性。
- **动作**：`action`，可用于鼓励能量最小或对称控制。

## 8. 不确定或不可用的信号
- **官方原始奖励**：被显式隐藏（masked），不可用。
- **显式成功/失败标志**：不存在。
- **存活步数/时间**：未直接提供在观察或 info 中，无法直接获取。
- **底座所处区域精细状态**（如是否接近边界）：只能通过底座位置计算，但位置是可用信号。
- **任何 info 中的额外奖励或终止原因**：info 为空，不存在。