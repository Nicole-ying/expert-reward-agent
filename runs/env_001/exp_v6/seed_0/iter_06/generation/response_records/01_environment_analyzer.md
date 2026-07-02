# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
控制一个具有主引擎与两侧姿态引擎的飞行体，从视口顶部中央附近出发，尽快降落到中央目标着陆平台上，并保持稳定停靠（身体不再处于活跃/awake状态）。整个过程中要尽量节省引擎推力，同时避免：
- 与地面或平台以外的区域发生碰撞/接触（crash）
- 水平位置超出视口边界
- 着陆时速度过大或姿态不稳（间接通过是否安全接触体现）

简言之：**以最少的燃料消耗**，**快速**降落到指定目标平台并**安全停稳**。

## 2. 任务类型选择
- **selected_route_id**: `navigation_goal_reaching`
- **confidence**: high
- **reason**: 任务核心是使飞行体从初始位置移动到指定目标平台并停靠，同时追求速度与消耗约束，具备明确的导航目标到达属性。虽然任务涉及燃料优化，但主要成功条件是到达并停稳于目标位置，与导航目标到达类型最匹配。

## 3. 观察空间 `observation_space`
- **type**: `Box`
- **shape**: `(8,)`
- **dtype**: `float32`（默认，未明确指定但符合连续量特征）
- **各维度含义**：

| 维度索引 | 名称 | 含义 |
|---------|------|------|
| 0 | `x_position` | 飞行体相对于目标平台中心的水平坐标 |
| 1 | `y_position` | 飞行体相对于平台高度的垂直坐标 |
| 2 | `x_velocity` | 水平方向线速度 |
| 3 | `y_velocity` | 垂直方向线速度 |
| 4 | `body_angle` | 机体倾斜角度（朝向） |
| 5 | `angular_velocity` | 机体旋转角速度 |
| 6 | `left_support_contact` | 左支撑腿是否接触（0.0/1.0） |
| 7 | `right_support_contact` | 右支撑腿是否接触（0.0/1.0） |

## 4. 动作空间 `action_space`
- **type**: `Discrete`
- **动作数量**: `4`
- **每个动作含义**：

| 动作ID | 名称 | 效果 |
|-------|------|------|
| 0 | `no_engine` | 不做任何推进，完全依赖当前动量与重力 |
| 1 | `left_orientation_engine` | 点燃左侧姿态引擎，产生旋转力矩（改变角度） |
| 2 | `main_engine` | 点燃主引擎，产生与机体朝向一致的推力 |
| 3 | `right_orientation_engine` | 点燃右侧姿态引擎，产生反向旋转力矩 |

## 5. step 与终止条件分析
### 5.1 终止模式
环境在任一以下条件成立时标记 `terminated=True`（且没有额外的截断条件，`truncated` 恒为 `False`）：

1. **crash_or_body_contact**：机体与地面或非目标区域发生接触/碰撞 → **失败型终止**
2. **horizontal_position_outside_viewport**：水平位置超出视口边界 → **失败型终止**
3. **body_not_awake_or_settled**：机体不再处于活跃/awake状态（例如停稳在平台上，或撞击后失稳停止） → **可能成功，也可能失败**

- **success-like termination**：仅由 `body_not_awake_or_settled` 触发，并且最终状态满足安全着陆条件（见下）时可视为成功。
- **failure-like termination**：`crash_or_body_contact`、`horizontal_position_outside_viewport`，以及 `body_not_awake_or_settled` 但不满足安全着陆条件时。
- **ambiguous termination**：仅有 `body_not_awake_or_settled` 本身属于模糊终止，必须结合观测进一步判断。
- **truncation**：本环境未使用截断（`truncated=False` 始终为假）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无明确 success 标记）
- **explicit_failure_flag_available**: false  
  （同样无明确 failure 标记）
- **allowed_info_fields**: 无（`info` 为空字典，不可依赖）
- **forbidden_or_uncertain_info_fields**: 所有字段，因为只返回空 `{}`。

> 因此，判断成功与否需要基于 `next_obs`（终止后的状态）自行推断安全着陆条件。典型安全着陆条件为：
> - 两个支撑腿均有接触（左右接触均为 1.0）
> - 水平位置接近 0（目标中心）
> - 垂直位置接近 0（已降到平台高度）
> - 机体角度接近 0（竖直）
> - 线速度与角速度很小

## 6. reward 函数接口契约
函数签名（此为下游接口规范，勿自行实现）：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用的输入**：
- `obs`：执行动作前的状态（8维向量）
- `action`：当前执行的动作ID（0~3）
- `next_obs`：执行动作后的状态（8维向量，终止时即为最终状态）
- `info`：空字典，不可使用
- `training_progress`：⚠️ 除非 prompt 明确说明可以使用，否则不要依赖

**严格禁止使用的输入**：
- `original_reward`：官方奖励已被掩码（masked），不可引用或还原
- 任何 `info` 中的字段（因为为空）
- 任何未声明的 `obs` 切片或环境内部变量

## 7. 可用于奖励函数的信号
以下信号可从 `obs` / `next_obs` 中安全提取，用于构造奖励：

- **位置信号**：
  - `x_position`（obs[0], next_obs[0]）—— 与目标平台的水平距离
  - `y_position`（obs[1], next_obs[1]）—— 与平台高度的垂直距离
- **速度信号**：
  - `x_velocity`（obs[2], next_obs[2]）
  - `y_velocity`（obs[3], next_obs[3]）
- **姿态信号**：
  - `body_angle`（obs[4], next_obs[4]）—— 机体倾斜角
  - `angular_velocity`（obs[5], next_obs[5]）
- **接触信号**：
  - `left_support_contact`（obs[6], next_obs[6]）
  - `right_support_contact`（obs[7], next_obs[7]）
- **动作/引擎信号**：
  - 当前动作 `action` 可用于惩罚引擎点火次数或主引擎使用

通过这些信号可以构建距离惩罚、速度惩罚、姿态惩罚、接触奖励，以及燃料消耗惩罚。

## 8. 不确定或不可用的信号
- **crash 细节**：环境只暴露了支撑腿的接触，未区分“安全接触”与“碰撞接触”（如机体其他部分触地）。因此，判断是否 crash 需要借助位置、速度、姿态及接触的组合逻辑，但无法直接从观测中获得绝对的 crash 标签。
- **官方奖励信息**：完全不可用。
- **info 中的任何成功/失败标记**：不存在。
