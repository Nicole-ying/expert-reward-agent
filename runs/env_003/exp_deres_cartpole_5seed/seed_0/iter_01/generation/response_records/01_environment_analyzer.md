# Response Record

# Env_003 环境理解卡片

## 1. 任务目标
智能体通过向左或向右施加一个固定大小的水平力，控制一个沿轨道移动的底座。目标是在底座不超出轨道边界（`|base_position| ≤ 2.4`）的前提下，尽可能保持底座上方未驱动的杆子直立（`|pole_angle| ≤ 0.20943951` 弧度）。任务以存活时间（步数）为核心追求，每步不倒塌、不出界即为成功延续，一旦违反任一终止条件即失败，在 500 步截断时视为完成一次完整尝试。

## 2. 任务类型选择
selected_route_id: survival_balance_task  
confidence: high  
reason: 任务核心是“保持平衡 + 存活”，没有显式目标位置或目标达成信号，只有失败边界。失败条件明确（杆子倾斜过大或底座出界），成功等同于存活至截断。这完全符合 survival_balance_task 的特征。

## 3. 观察空间 observation_space
- type: Box (连续值)
- shape: (4,)
- dtype: float32
- obs[0] (index 0, base_position): 底座在轨道上的水平位置，理论范围 [-4.8, 4.8]，终止边界为 ±2.4。
- obs[1] (index 1, base_velocity): 底座的水平速度，无界。
- obs[2] (index 2, pole_angle): 杆子相对于竖直方向的角度（弧度），范围 [-0.418879, 0.418879]，实际终止边界为 ±0.20943951。
- obs[3] (index 3, pole_angular_velocity): 杆子的角速度，无界。

## 4. 动作空间 action_space
- type: Discrete(2)
- action 0: push_negative_direction —— 向轨道负方向施加固定水平力。
- action 1: push_positive_direction —— 向轨道正方向施加固定水平力。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无。没有“到达目标”等成功事件，唯一正向结果是存活到 500 步截断。
- failure-like termination: 底座位置绝对值超过 2.4（出界），或杆子角度绝对值超过 0.20943951（倒下），二者之一发生即终止，属于失败。
- ambiguous termination: 无。
- truncation: 达到 500 步时截断，此时底座和杆子均未触发失败条件，可视为一次完整的存活尝试。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: `info` 固定为 `{}`，此环境中没有任何额外字段可用。
- forbidden_or_uncertain_info_fields: 所有自定义字段（如 `"success"`, `"failure"`, `"termination_reason"` 等）均不存在且禁止使用。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前步的观察（shape (4,)）
- `action`：执行的动作（0 或 1）
- `next_obs`：下一步的观察（shape (4,)）
- `info`：仅限当前环境明确暴露的字段（当前为空字典）

严格禁止：
- `original_reward`：官方奖励已被隐藏，不得任何形式使用或还原。
- 任何未在 `allowed_info_fields` 中列出的 `info` 字段。
- `training_progress` 参数（本任务描述中未明确允许使用它）。

## 7. 可用于奖励函数的信号
以下信号全部来自 `obs` 和 `next_obs`，无需依赖 `info`：
- 底座位置：`obs[0]`, `next_obs[0]`
- 底座速度：`obs[1]`, `next_obs[1]`
- 杆子角度：`obs[2]`, `next_obs[2]`（越接近 0 越直立）
- 杆子角速度：`obs[3]`, `next_obs[3]`
- 动作：`action`（推力方向）

## 8. 不确定或不可用的信号
- 官方原始奖励：已遮蔽，禁止使用。
- 任何 `info` 内的自定义标志（如 `"success"`, `"failure"`, `"TimeLimit.truncated"` 等）：不存在且不可用。
- 额外的环境信息（如力矩、能量消耗等）：未提供，不可用。
