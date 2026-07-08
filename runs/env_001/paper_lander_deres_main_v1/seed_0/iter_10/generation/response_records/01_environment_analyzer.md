# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
一个二维飞行器从视口上方初始位置出发，受随机初始力。目标是 **尽快飞抵中央目标着陆点**，并在其上 **平稳停靠**（速度降至零、保持竖直姿态、双腿触地），同时尽可能 **少用主引擎推力** 以节省燃料。朝向、接触、越界等都会影响终止。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 推测为 float32（位置速度等为连续值，接触为 0/1，但整体为浮点）
- obs[0]: x_position — 相对于目标着陆点中心的水平坐标
- obs[1]: y_position — 相对于目标着陆点高度平面的垂直坐标
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体倾角（弧度制，推测）
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左支撑点/腿接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact — 右支撑点/腿接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不开启任何引擎，滑行
- action 1: left_orientation_engine — 开启左侧姿态引擎（产生旋转力矩）
- action 2: main_engine — 开启主引擎（产生向上的推力）
- action 3: right_orientation_engine — 开启右侧姿态引擎（产生相反方向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**: body_not_awake_or_settled 且满足成功条件（靠近目标、低速、竖直、双腿触地）——即平稳停靠目标上。
- **failure-like termination**: crash_or_body_contact（机体或关键部位强烈撞击地面/结构），horizontal_position_outside_viewport（水平飞出边界）。
- **ambiguous termination**: body_not_awake_or_settled 但未完全满足目标位置或接触条件（可能悬空失速、摔倒等），需根据上下文判断。
- **truncation**: 本环境未明确提及最大步数截断，但通常 RL 框架有步数上限，属于截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 返回空字典 {}）
- forbidden_or_uncertain_info_fields: info 中任何字段均不可用（不存在 success、failure、termination_reason 等）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- `obs`（当前观察）
- `action`（刚执行的动作）
- `next_obs`（下一时刻观察）
- `info` 为空，所以实际上没有可用 info 字段
- `training_progress` 只有在明确需要时才使用，此处任务未要求，故不应依赖

禁止使用：
- `original_reward`（官方奖励，严禁读取）
- `official_reward` 或其别名
- 未在观察空间中声明的 `obs` 切片
- 未明确允许的 `info` 字段（均不允许）

## 7. 可用于奖励函数的信号
- **position**: next_obs[0]（x 向目标距离）, next_obs[1]（y 向目标高度差）
- **velocity**: next_obs[2]（水平速度）, next_obs[3]（垂直速度）
- **orientation**: next_obs[4]（倾角）, next_obs[5]（角速度）
- **contact**: next_obs[6]（左腿触地）, next_obs[7]（右腿触地）
- **action/engine**: 使用 action == 2（主引擎）可计入燃料惩罚

## 8. 不确定或不可用的信号
- 是否成功着陆无显式标记，必须结合终止时刻的 `next_obs` 自行判断（如距离小、速度低、竖直、双腿触地）。
- 越界、坠毁等失败终止也只能从位置、速度突变或终止时刻的位置推断，但无法直接获得“crash”标志。
- 无法获得“是否静止”的直接信号，但可通过速度大小和角速度推断。
- 无 `info` 字段可用，无法获得任何环境内部度量（如剩余燃料、当前冲击力等）。
