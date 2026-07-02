# Env_001 环境理解卡片

## 1. 任务目标
一个类似飞行器的物体从视口顶部中央附近开始运动（有初始随机力），目标是**尽快到达并稳定停留在画面中央的目标着陆垫上**，同时**尽可能少地使用引擎推力**。
智能体需要学习：
- 向目标点靠近（调整位置和速度），
- 在到达目标区域时将速度减到很小（“settle”），
- 保持平稳的姿态（角度接近零），
- 安全着陆（左右支撑脚都与垫子接触）。
整个过程在时间和燃油消耗之间寻求平衡。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 核心目标是“让智能体到达并稳定在指定的目标位置上”，属于典型的导航/目标到达任务。最低消耗和快速完成是对到达策略的附加约束，并不改变任务的根本性质。

## 3. 观察空间 observation_space
- type: `Box`
- shape: `(8,)`
- dtype: 提取的特征均为 `float32`（接触标志为 0.0 或 1.0）
- 各维度含义：

| 索引 | 名称                      | 含义与说明 |
|------|---------------------------|------------|
| 0    | x_position                | 机体水平坐标，相对于目标着陆垫的横向偏移 |
| 1    | y_position                | 机体垂直坐标，相对于着陆垫高度的偏移 |
| 2    | x_velocity                | 机体水平方向线速度 |
| 3    | y_velocity                | 机体竖直方向线速度 |
| 4    | body_angle                | 机体俯仰/倾斜角度（例如与水平面的夹角） |
| 5    | angular_velocity          | 机体角速度 |
| 6    | left_support_contact      | 左侧支撑/起落架是否接触目标垫，接触为 1.0，否则 0.0 |
| 7    | right_support_contact     | 右侧支撑/起落架是否接触目标垫，接触为 1.0，否则 0.0 |

## 4. 动作空间 action_space
- type: `Discrete(4)`（共 4 种动作）
- 每个动作的含义：

| action id | 名称                         | 含义 |
|-----------|------------------------------|------|
| 0         | no_engine                    | 不点火，不施加任何推力 |
| 1         | left_orientation_engine      | 点燃左侧姿态调节引擎（产生一个旋转力矩） |
| 2         | main_engine                  | 点燃主引擎（产生向上的推力） |
| 3         | right_orientation_engine     | 点燃右侧姿态调节引擎（产生与动作 1 方向相反的旋转力矩） |

本质上是一个**执行器集合**：不动、左转、加推力、右转。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  body_not_awake_or_settled（机体稳定或不再活动） **有可能** 意味着成功——如果此时机体在目标垫上、接触标志均为 1、速度极小、角度接近 0，可以判定为成功着陆。但该条件本身只是“身体沉睡或稳定”，也可能在垫子外发生（坠落后静止），因此不能直接等同于成功。
- **failure-like termination**:  
  - crash_or_body_contact（除目标垫以外的碰撞/身体接触）——大概率是与其他结构或地面发生撞击，非安全着陆。  
  - horizontal_position_outside_viewport（水平位置超出视口）——机体飞出画面，显然失败。
- **ambiguous termination**: body_not_awake_or_settled 在接触标志为 0 或位置远离原点时，也可能是失败（卡在角落静止）。
- **truncation**: masked step source 中未见时间/步数截断参数，但环境可能设有最大步数（常见），本卡片不依赖该信息。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: (空，step 返回的 info 为空字典 `{}`)
- forbidden_or_uncertain_info_fields: 禁止假设 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等字段存在。

**结论**：必须通过 `obs` 和 `next_obs` 中的位置、速度、接触标志等信息自行推断成功或失败。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`（上一时刻观测）
- `action`（刚执行的动作）
- `next_obs`（新观测）
- `info` 中的字段（当前 info 为空，故无可用）
- `training_progress` **仅在 prompt 明确指示允许使用时才可用**，当前任务未提及，因此一律视为**禁止使用**。

**禁止使用**：
- `original_reward`（即官方原始奖励，被 mask 隐藏）
- 任何称为 `official_reward` 或从环境中直接读取的奖励值
- `info` 中不存在或未声明的任何键
- 观测中未声明的切片或编码

## 7. 可用于奖励函数的信号
可以从观测、动作和下一观测中提取以下有意义的信号：

- **position**: `next_obs[0]`（x位置）、`next_obs[1]`（y位置）——描述机体相对于目标垫的位置；距离目标越近越好。
- **velocity**: `next_obs[2]`（水平速度）、`next_obs[3]`（垂直速度）——用于鼓励低速稳定着陆，或在飞行过程中保持适当速度。
- **orientation**: `next_obs[4]`（机体角度）——目标角度应为 0（水平），角度小有利于着陆。
- **angular_velocity**: `next_obs[5]`——小角速度表明姿态稳定。
- **contact**: `next_obs[6]` 和 `next_obs[7]`——两侧接触标志。两者均为 1.0 且位置、速度接近零时，可视为成功着陆的强烈信号。
- **action/engine**: `action` 自身——可用于衡量推进器使用情况（非 0 动作代表消耗能量/燃料）。例如惩罚使用主引擎或姿态引擎。

## 8. 不确定或不可用的信号
- **显式成功标志**：无，不可从 info 获取。
- **显式失败标志**：无，不可从 info 获取。
- **crash/碰撞的具体对象或接触力信息**：不包含在观测中，无法直接获知碰撞类型，只能通过位置、接触标志间接推断。
- **剩余的步数或时间限制信息**：观测中未提供，不可用。
- **环境内部的燃料/能量计算**：未被暴露，仅能通过动作使用的引擎类型近似衡量。
- **水平或垂直方向的绝对坐标（世界坐标）**：观测给出的是相对目标垫的偏移量，只能获取相对位置，不能知道绝对范围边界。
- **任何与 map/terrain 相关的传感器信息**：观测中无，不可用。