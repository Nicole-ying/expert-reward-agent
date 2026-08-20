# Env_001 环境理解卡片

## 1. 任务目标
这是一个二维飞行器轨迹优化任务。  
飞行器从视野顶部中央附近出发，带有随机的初始作用力。  
**核心目标**：尽快抵达并稳定停在中央的目标平台上。  
**附加要求**：尽量减少发动机推力的使用。  
智能体需要学会**靠近目标、减速、保持平稳姿态并安全接触平台**。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务描述明确指出“到达并稳定在中央目标平台”，属于典型的导航目标到达任务，推力最小化是附加的效率约束，整体核心仍是达到指定目标状态。

## 3. 观察空间 observation_space
- **type**: Box
- **shape**: (8,)
- **dtype**: float32 （接触标志虽为 0/1，但仍以 float 存储）
- **各维度含义**：
  - `obs[0]`: **x_position** — 飞行器相对于目标平台的水平坐标
  - `obs[1]`: **y_position** — 飞行器相对于平台高度的垂直坐标
  - `obs[2]`: **x_velocity** — 水平线速度
  - `obs[3]`: **y_velocity** — 垂直线速度
  - `obs[4]`: **body_angle** — 机体朝向角
  - `obs[5]`: **angular_velocity** — 角速度
  - `obs[6]`: **left_support_contact** — 左支撑接触标志（0.0 或 1.0）
  - `obs[7]`: **right_support_contact** — 右支撑接触标志（0.0 或 1.0）

## 4. 动作空间 action_space
- **type**: Discrete
- **动作数量**: 4
- **每个动作的含义**：
  - `action 0` : **no_engine** — 不执行任何发动机操作（无推力）
  - `action 1` : **left_orientation_engine** — 点燃一个方向发动机，产生（例如）逆时针调整力矩
  - `action 2` : **main_engine** — 点燃主发动机，产生向前推力
  - `action 3` : **right_orientation_engine** — 点燃相反的方向发动机，产生反向调整力矩

## 5. step 与终止条件分析

### 5.1 终止模式
环境在满足以下任一条件时返回 `terminated = True`：
- **crash_or_body_contact**（碰撞或机体接触）：可能对应损坏或意外接触，视为 **failure‑like termination**。
- **horizontal_position_outside_viewport**（水平位置超出视野）：飞行器飞出有效区域，视为 **failure‑like termination**。
- **body_not_awake_or_settled**（机体不再活跃或已稳定）：若满足稳定条件（例如靠近目标且速度很小）则可视为 **success‑like termination**；若因失控失能也可能为失败，但多数实现中作为成功着陆信号，**本环境未在终止描述中显式区分**，因此带有 **ambiguous termination** 的性质。
此外，`truncation` 始终为 `False`，没有最大步数截断。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 无（`info` 字典始终为空 `{}`）
- **forbidden_or_uncertain_info_fields**: 所有 `info` 字段均禁止使用（因为未声明任何可用字段）；禁止假设存在 `info["success"]`、`info["failure"]` 等信号。

## 6. reward 函数接口契约

函数签名如下（参数名已固定）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs` — 当前步观察（8 维数组）
- `action` — 所选动作（整数）
- `next_obs` — 下一时刻观察（8 维数组）
- `info` — 仅能使用**空字典**，禁止读取任何字段
- `training_progress` — **只有 prompt 中明确说明可用时才允许使用**；本任务描述未提及，故默认禁止

**禁止使用**：
- `original_reward` — 不得直接或间接复制官方隐藏奖励
- 任何未在“允许使用”中列出的 `info` 字段
- 任何未经声明的 `obs` 切片

## 7. 可用于奖励函数的信号
基于观察和动作，可以直接访问以下信号（均为合法输入）：
- **位置**：`obs[0]`（x 相对坐标），`obs[1]`（y 相对坐标）—— 可计算与目标的距离
- **速度**：`obs[2]`（水平速度），`obs[3]`（垂直速度）—— 可用于速度惩罚
- **姿态**：`obs[4]`（机体角度）—— 可用于姿态稳定奖励
- **角速度**：`obs[5]`—— 可惩罚不必要的旋转
- **接触**：`obs[6]`, `obs[7]`（左右支撑接触）—— 可指示着陆状态
- **动作/发动机**：`action` 取值—— 可区分无推力、调姿、主推力，用于能耗惩罚

## 8. 不确定或不可用的信号
- **明确成功/失败标志**：不存在（`info` 为空，`terminated` 仅给出布尔且不区分原因）
- **目标达成判断**：无法从单一布尔值直接获得，需要结合 `next_obs` 位置、速度、接触等猜测是否成功
- **任何 `info` 内部字段**：全部不可用
- **`original_reward`**：严格禁止使用
- **`training_progress`**：除非 prompt 额外授权，否则不可用