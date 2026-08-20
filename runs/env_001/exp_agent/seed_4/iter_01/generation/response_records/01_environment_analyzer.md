# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
- 控制一个 2D 飞行器（或类车辆体），从其初始位置（视口顶部中心附近，带随机初始力）移动到画面中央的圆形目标垫上，并稳定停靠。
- 任务同时要求**尽快完成**（路径时间短）且**尽量减少发动机推力使用**（省油）。
- 理想行为：飞行器接近目标垫、减速、保持正确姿态（角度接近垂直）、平稳触垫且不再弹起或滑动。

## 2. 任务类型选择
- selected_route_id: `navigation_goal_reaching`
- confidence: high
- reason: 核心是到达一个明确的目标位置（中央目标垫）并稳定停靠，且需要路径优化（速度、姿态）。虽然包含推力最小化的子目标，但本质仍为带约束的有目标导航任务，而非纯生存平衡或多目标强化学习。

## 3. 观察空间 observation_space
- type: Box (连续值向量)
- shape: (8,)
- dtype: float32（典型值）
- obs[0]: **x_position** —— 飞行器相对于目标垫中心的水平坐标。
- obs[1]: **y_position** —— 飞行器相对于目标垫高度的垂直坐标（向上为正）。
- obs[2]: **x_velocity** —— 水平线性速度。
- obs[3]: **y_velocity** —— 垂直线性速度。
- obs[4]: **body_angle** —— 飞行器本体的倾斜角（弧度，0 表示垂直向上）。
- obs[5]: **angular_velocity** —— 角速度。
- obs[6]: **left_support_contact** —— 左支撑脚/腿与地面（或垫子）接触标志（1.0 表示接触）。
- obs[7]: **right_support_contact** —— 右支撑脚/腿与地面接触标志（1.0 表示接触）。

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: **no_engine** —— 不开启任何引擎，飞行器仅受重力和惯性影响。
- action 1: **left_orientation_engine** —— 点燃左侧姿态调整引擎，产生逆时针旋转力矩（通常使角速度正向增加）。
- action 2: **main_engine** —— 点燃主引擎，沿飞行器当前朝向方向产生推力，使飞行器加速。
- action 3: **right_orientation_engine** —— 点燃右侧姿态调整引擎，产生顺时针旋转力矩（通常使角速度负向增加）。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**:  
  `body_not_awake_or_settled` —— 当飞行器长时间无运动（物理引擎判定为 asleep）或已在地面稳定静止时触发。如果此时飞行器位于目标垫上且姿态合理，则可视为成功着陆；但若停在错误位置（例如视口边缘地面）也属于该终止，因此单独看并不可靠。
- **failure-like termination**:  
  `crash_or_body_contact` —— 飞行器主体（非支撑脚）触碰地面或其它障碍，代表坠毁；  
  `horizontal_position_outside_viewport` —— 水平位置越界，代表飞出允许区域。
- **ambiguous termination**:  
  `body_not_awake_or_settled`（如上），需要结合位置信息（obs[0], obs[1]）来判断是否真正成功。
- **truncation**:  
  源码中返回的第四个值为 `False`，说明环境没有设置时间上限截断（或未显式给出），但可能的 `training_progress` 参数未在此环境中引入。因此 **无显式 truncation**。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  原因：`step` 返回的 info 为 `{}`，没有任何 `success`、`failure` 或 `termination_reason` 字段。
- explicit_failure_flag_available: false  
  同上。
- allowed_info_fields: 无（info 为空字典，所以无可用字段）。
- forbidden_or_uncertain_info_fields:  
  - `original_reward` / `official_reward` (禁止使用)  
  - 任何未在观察空间或明确授权中出现的键（例如假定存在的 `"success"` 等）均为危险且不可用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`（当前观察的 8 维向量）
- `action`（当前执行的动作 ID）
- `next_obs`（下一时刻的 8 维向量）
- `info` 中**明确允许的字段**（当前环境下无任何允许字段）
- `training_progress` **仅在 prompt 明确要求使用时可用**（本环境未提供，禁止使用）

禁止使用：
- `original_reward`（已被掩码，严禁以任何方式利用或近似反推）
- `official_reward` 或任何变体
- 未声明的 `info` 字段
- 未声明的 `obs` 切片含义（例如不能假设 obs[6] 和 obs[7] 之外还有其他接触信息）

## 7. 可用于奖励函数的信号
- **position**: `next_obs[0]` 和 `next_obs[1]` 提供相对于目标垫的水平与垂直距离，可直接用于距离导向奖励。
- **velocity**: `next_obs[2]` 和 `next_obs[3]` 可用于鼓励在接近目标时减速，或惩罚高速撞击。
- **orientation**: `next_obs[4]` 和 `next_obs[5]` 可用于惩罚倾斜角过大（不垂直）或高速旋转。
- **contact**: `next_obs[6]` 和 `next_obs[7]` 表示支撑脚是否触地，可用于检测着陆状态，需结合位置判断是否为成功着陆。
- **action/engine**: `action` 可用于惩罚引擎使用（动作1,2,3）以达成省油目标。

## 8. 不确定或不可用的信号
- **explicit success / failure flags**: info 完全为空，不能直接判定任务成功与否，必须通过位置、速度、接触等自身信息推断。
- **original_reward**: 掩码不可用。
- **training_progress**: 未提供，不可用。
- **任何未声明的物理量**（如剩余燃料、力矩大小）均不可假设存在。
