# 匿名环境理解卡片

## 1. 任务目标
本环境是一个2D类车辆轨迹优化任务。一个主体从视口顶部中央附近开始，带有随机初始力。目标是**尽快到达并稳定在中央目标垫上**，同时**尽可能少地使用引擎推力**。智能体需要学会靠近目标、减速、保持稳定姿态并安全着陆。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position — 相对于目标垫的水平坐标
- obs[1]: y_position — 相对于目标垫高度的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 身体朝向角
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑接触标志（0.0 或 1.0）
- obs[7]: right_support_contact — 右支撑接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不施加任何引擎推力
- action 1: left_orientation_engine — 启动左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 启动主引擎（产生主要推力）
- action 3: right_orientation_engine — 启动右侧姿态引擎（产生反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
环境在以下任一条件成立时终止（terminated = True）：
- `crash_or_body_contact` — 发生撞击或非安全接触
- `horizontal_position_outside_viewport` — 水平坐标超出视口范围
- `body_not_awake_or_settled` — 身体进入休眠状态或完全静止

分析：
- 前两条通常对应失败（撞击、飞出边界）。
- `body_not_awake_or_settled` 含义模糊：当主体在目标垫上成功稳定着陆时，也会触发静止（成功情况）；但主体可能在错误位置停止或卡住，同样触发。因此该终止**不能直接区分成功与失败**。
- 所有终止信息仅合并为一个布尔值 `terminated`，无额外解释字段。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: info 的所有字段（不可用），original_reward，任何未在 observation 中声明的信号

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` — 当前观测向量
- `action` — 当前动作索引
- `next_obs` — 下一步观测向量
- 无

禁止使用：
- `original_reward`（官方奖励已掩码，不得还原）
- `info` 的任何字段（info 为空，且不允许假设其存在字段）
- `training_progress`（本环境说明中未允许使用）
- 任何未在 observation_space 中明确声明的观测切片或计算

## 7. 可用于奖励函数的信号
以下信号完全从 `obs`、`action`、`next_obs` 中可获取：
- **位置信号**：x_position, y_position（相对于目标垫），可用于度量到目标的距离
- **速度信号**：x_velocity, y_velocity，可用于惩罚过快运动或鼓励减速
- **姿态信号**：body_angle, angular_velocity，可用于鼓励保持稳定角度（例如竖直）
- **接触信号**：left_support_contact, right_support_contact，可反映着陆脚是否着地、是否平稳
- **动作/引擎信号**：action 选择（0~3），可用于惩罚非零动作以节省燃料

## 8. 不确定或不可用的信号
- **显式成功/失败标志**：不可用，终止逻辑中不提供 success/failure 标记
- **终止原因解析**：无法准确得知是因为安全着陆还是异常停止而终止
- **官方原始奖励**：被屏蔽，禁止使用
- **任何 info 字段**：空字典，不可用
- **真实环境名或基准信息**：已匿名化，不可使用