# Env_001 环境理解卡片

## 1. 任务目标
一个2D物体（类似可悬停飞行器）从画布顶部中心附近出发，受到一个随机初始力。  
任务目标是**尽快到达中央的目标平台并稳定停靠**，同时**尽可能少地使用引擎推力**。  
智能体需要学会靠近目标、减速、保持稳定姿态，并实现安全的接触着陆。

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务明确要求同时优化多个目标——快速到达目标、最小化推力消耗、保持稳定方向、确保安全接触，属于典型的多目标权衡问题。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推测，基于连续物理量）
- 各维度含义（按 index 顺序）：
  - obs[0]: x_position — 相对于目标平台中心的水平坐标
  - obs[1]: y_position — 相对于目标平台高度的垂直坐标
  - obs[2]: x_velocity — 水平线速度
  - obs[3]: y_velocity — 垂直线速度
  - obs[4]: body_angle — 机体方向角
  - obs[5]: angular_velocity — 角速度
  - obs[6]: left_support_contact — 左侧支撑接触标志（1.0 表示接触，0.0 表示未接触）
  - obs[7]: right_support_contact — 右侧支撑接触标志（1.0 表示接触，0.0 表示未接触）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作编号与含义：
  - action 0: no_engine（不点火，什么也不做）
  - action 1: left_orientation_engine（点燃左侧姿态引擎，产生右转力矩/推力分量）
  - action 2: main_engine（点燃主引擎，产生主要上升/减速推力）
  - action 3: right_orientation_engine（点燃右侧姿态引擎，产生左转力矩/推力分量）

## 5. step 与终止条件分析

### 5.1 终止模式
step 内部的终止信号由以下三个条件任意满足时触发（terminated = True）：
- **crash_or_body_contact**: 发生碰撞或与不应接触的表面接触 → **失败型终止**
- **horizontal_position_outside_viewport**: 水平坐标超出视野范围 → **失败型终止**
- **body_not_awake_or_settled**: 物体不再活跃或已稳定 → **模糊型终止**（可能是成功停靠，也可能是坠毁后静止或其他非理想静止）

未观察到显式的截断（truncation）信号（返回 `truncated=False`），因此本环境在分析时不考虑最大步数截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  原环境 step 返回的 info 为空字典，不包含任何成功标志字段。
- explicit_failure_flag_available: false  
  同上，没有失败标志字段。
- allowed_info_fields: 无（info 为空，没有可用字段）
- forbidden_or_uncertain_info_fields: 所有可能存在的 info 字段均不可用（如 `success`, `failure`, `termination_reason` 等均不存在）。

> 注意：不能假设 info 中存在 `info["success"]` 或 `info["failure"]` 等字段用于奖励计算。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用的输入：**
- `obs`（当前观测）
- `action`（当前执行的动作）
- `next_obs`（下一观测）
- 以及 `info` 中被明确允许的字段（本环境中 `info` 为空，故不使用）

**严格禁止使用的输入：**
- `original_reward`（或任何形式的官方奖励）
- `info` 中未明确允许的任何字段（即全部字段）
- `training_progress`（除非 prompt 明确要求使用，本环境中未被要求）
- 任何对观测的未声明的隐含切片或解释（需基于已声明的字段含义使用）

## 7. 可用于奖励函数的信号
以下信号均可由 `obs` 和 `next_obs` 安全构造，并用于设计奖励分量（但奖励本身不由本卡片设计）：
- **位置偏差**：水平位置 `obs[0]`（或 `next_obs[0]`），垂直高度 `obs[1]`。
- **速度**：水平速度 `obs[2]`，垂直速度 `obs[3]`，可合成合速度大小。
- **姿态**：机体角度 `obs[4]`（偏离竖直或目标方向的角度）。
- **角速度**：`obs[5]`。
- **接触信号**：左接触 `obs[6]`，右接触 `obs[7]`；可用于判断是否着陆、是否双足平稳接触。
- **动作/发动机使用**：动作编号（0~3）本身，可用于衡量推力使用频率（如主引擎使用惩罚）。

衍生的复合信号举例（可不直接列出，此处仅为澄清）：
- 到目标点的欧氏距离：基于 obs[0], obs[1]。
- 速度大小、加速度变化（通过 obs 与 next_obs 速度差近似）。
- 稳定性指标：低速度 + 低角速度 + 双接触标志同时为 1 + 小角度偏差。

## 8. 不确定或不可用的信号
- **绝对成功/失败标志**：无法从 info 中获得，必须通过观测自构建判断逻辑。
- **目标平台的具体边界坐标**：仅知道“中心”和“平台高度”的参考系，但未给出容差范围，需要在奖励中保守假设或通过学习推断。
- **燃油存量或剩余能量**：观测中不包含该量，只能通过动作序列间接估计推力消耗，但不能获得精确能耗。
- **原环境内置奖励**：被屏蔽，严禁使用。
- **任何额外 info 字段**：如 None、未出现的键值均不可假设存在或可用。