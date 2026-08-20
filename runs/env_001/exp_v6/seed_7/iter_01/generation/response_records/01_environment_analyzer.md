# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
被匿名化的环境是一个 2D 类飞行器轨迹优化任务。  
主体从视口顶部中央附近出发，带有随机初始力。  
目标是在尽可能短的时间内到达并稳定降落在中央的目标着陆垫上，同时尽可能少地使用引擎推力。  
智能体需要学会靠近目标、减小速度、保持稳定的姿态并实现安全的接触着陆。

---

## 2. 任务类型选择
selected_route_id: multi_objective_task  
confidence: high  
reason: 任务描述明确要求平衡“到达速度”与“引擎使用最少”两个目标，属于典型的多目标优化问题（快速 vs 燃料节约）。

---

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float (通常为 float32)
- obs[0]: x_position — 横向坐标，相对于目标着陆垫中心的水平距离
- obs[1]: y_position — 竖直坐标，相对于着陆垫高度的垂直距离
- obs[2]: x_velocity — 水平线速度
- obs[3]: y_velocity — 垂直线速度
- obs[4]: body_angle — 机体朝向角度
- obs[5]: angular_velocity — 角速度
- obs[6]: left_support_contact — 左侧支撑腿接触标志，1.0 接触，0.0 未接触
- obs[7]: right_support_contact — 右侧支撑腿接触标志，1.0 接触，0.0 未接触

---

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine — 不点火，保持惯性
- action 1: left_orientation_engine — 启动左姿态微调引擎
- action 2: main_engine — 启动主引擎（产生主推力）
- action 3: right_orientation_engine — 启动右姿态微调引擎

---

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：环境中未提供显式成功标志。当触发 `body_not_awake_or_settled` 且智能体已稳定着陆在垫上（位置接近0、速度极小、两侧支撑腿接触）时，可视为成功。
- **failure-like termination**：
  - `crash_or_body_contact`：与非垫面或障碍物发生碰撞，或机体与地面猛烈接触，视为失败。
  - `horizontal_position_outside_viewport`：水平位置超出边界，视为失败。
- **ambiguous termination**：
  - `body_not_awake_or_settled` 本身可能是成功着陆后的稳定状态，也可能是在空中失去动力后掉落到无效区域后的静止。需结合观测（位置、接触）才能区分。
- **truncation**：step 返回的 truncated 值为 `False`，无时间限制截断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: （无，step 返回的 info 为空字典 `{}`）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，尤其禁止假设 `info["success"]`、`info["failure"]` 或类似键存在。

---

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

允许使用：
- obs：当前步的完整观测向量
- action：当前步执行的动作
- next_obs：执行动作后的观测向量
- training_progress：仅在环境 prompt 明确允许时使用（这里未声明，故不推荐依赖）

禁止使用：
- original_reward：已被掩码的官方奖励，严禁直接或间接使用
- info：当前为空字典，不可假设任何内容
- 未声明的 obs 切片或隐藏状态

---

## 7. 可用于奖励函数的信号
- position: obs[0] (x_position), obs[1] (y_position)，next_obs[0], next_obs[1]
- velocity: obs[2] (x_velocity), obs[3] (y_velocity)，及其 next_obs 对应项
- orientation: obs[4] (body_angle), obs[5] (angular_velocity)，及其 next_obs 对应项
- contact: obs[6] (left_contact), obs[7] (right_contact)，及其 next_obs 对应项
- action/engine: action 取值（0/1/2/3），可用于估算推力或燃料消耗

---

## 8. 不确定或不可用的信号
- 官方奖励 original_reward（被完全掩码，不可用）
- info 字典中的所有字段（当前为空，无成功/失败/原因等信息）
- 任何未在观察空间注明的内部物理量（如引擎推力大小、燃料剩余等）
- 外部地图或绝对世界坐标（观察仅提供相对目标垫的信息）
