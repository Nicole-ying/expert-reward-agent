# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个两足身体在不平坦地形上尽可能远、尽可能快地向前行走，同时尽量减小能量消耗。智能体需要协调两条腿的髋关节和膝关节，产生稳定的步态。如果身体摔倒，回合立即结束；如果走到地形尽头，也会终止。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (24,)
- dtype: float32
- obs[0]: hull_angle — 身体相对于竖直方向的倾角（弧度）
- obs[1]: hull_angular_velocity — 身体的角速度（弧度/秒）
- obs[2]: horizontal_velocity — 前向/后向的线速度（米/秒）
- obs[3]: vertical_velocity — 上下方向的线速度（米/秒）
- obs[4]: hip1_angle — 腿1的髋关节角度（弧度）
- obs[5]: hip1_speed — 腿1的髋关节角速度（弧度/秒）
- obs[6]: knee1_angle — 腿1的膝关节角度（弧度）
- obs[7]: knee1_speed — 腿1的膝关节角速度（弧度/秒）
- obs[8]: leg1_contact — 腿1与地面的接触标志（1.0 接触，0.0 未接触）
- obs[9]: hip2_angle — 腿2的髋关节角度（弧度）
- obs[10]: hip2_speed — 腿2的髋关节角速度（弧度/秒）
- obs[11]: knee2_angle — 腿2的膝关节角度（弧度）
- obs[12]: knee2_speed — 腿2的膝关节角速度（弧度/秒）
- obs[13]: leg2_contact — 腿2与地面的接触标志（1.0 接触，0.0 未接触）
- obs[14]~obs[23]: lidar_0~lidar_9 — 10个激光雷达测距仪沿地形前方的距离测量值（米或其他线性单位）

## 4. 动作空间 action_space
- type: Box
- shape: (4,)
- 每个维度连续，值域 [-1.0, 1.0]
- action[0]: hip_torque_leg1 — 施加在腿1髋关节上的力矩
- action[1]: knee_torque_leg1 — 施加在腿1膝关节上的力矩
- action[2]: hip_torque_leg2 — 施加在腿2髋关节上的力矩
- action[3]: knee_torque_leg2 — 施加在腿2膝关节上的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain` — 智能体成功走完整个地形
- failure-like termination: `body_fallen_over` — 身体摔倒（可能是倾斜过度或接触异常）
- ambiguous termination: 无
- truncation: 无（`step` 返回的第二个终止标志恒为 False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为 info 为空）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs：当前观测数组（可整体使用，或按上述索引提取）
- action：当前执行的动作数组
- next_obs：下一步观测数组
- info：仅当环境明确提供且经分析允许的字段，这里为空字典，无法提供额外信号
- training_progress：禁止使用（prompt 未明确要求使用）

禁止使用：
- original_reward（官方奖励已被屏蔽）
- official_reward
- 任何未在 observation_space 字段中明确声明的 obs 切片
- 任何 info 字段（info 为空且无保证）

## 7. 可用于奖励函数的信号
- position: 没有直接的世界坐标，但可通过速度积分推断前进位移（如累计水平速度）
- velocity: horizontal_velocity（前向速度），鼓励快速前进；vertical_velocity（垂直速度，过大的垂直振荡可能代表不稳定）
- orientation: hull_angle（身体倾角，接近0表示稳定），hull_angular_velocity（角速度，用于惩罚剧烈摇晃）
- contact: leg1_contact, leg2_contact（脚掌着地信息，可用于奖励合理的步态转换或避免长时间双脚离地/单脚着地）
- action/engine: 动作幅度（如 sum(|action|) 或 sum(action^2)）可用于惩罚过大力矩以减少能量消耗
- lidar: lidar_0..9（前方地形距离，作为潜能信号鼓励前进；但需小心使用，避免只依赖前方空旷而不鼓励实际前进）

## 8. 不确定或不可用的信号
- 世界坐标/绝对位置：未提供，无法直接获取“已行走距离”。
- 成功标志：info 为空，无显式 success/failure 标志，只能通过终止条件推断（但终止条件本身不直接出现在 reward 中）。
- 地形尽头到达信号：仅在终止时触发，不能在 reward 中预测使用，但可用于事后分析。
- 能耗的精确物理量：没有直接的能耗观测，只能通过动作幅度近似。
- 地形难度/坡度：未显式提供，只能通过 lidar 间接感知。
