# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
控制一个位于画面顶部中心附近的 2D 飞行器，在随机初始推力作用下，**快速且稳定地降落到中央目标平台**，同时**尽量节省引擎燃料**。  
理想表现是：平稳接近目标、减速、保持小角度、双腿安全接触平台并让身体停稳。

---

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

---

## 3. 观察空间 observation_space
- **type**: Box  
- **shape**: (8,)  
- **dtype**: float32（推断）  

各维度含义（均为相对值）：
- **obs[0]** – `x_position`：飞行器质心相对目标平台中心的水平坐标（目标中心为 0）
- **obs[1]** – `y_position`：飞行器质心相对目标平台高度的垂直坐标（平台高度为 0）
- **obs[2]** – `x_velocity`：水平线速度
- **obs[3]** – `y_velocity`：竖直线速度
- **obs[4]** – `body_angle`：机身朝向角（弧度或度，通常 0 表示竖直向上）
- **obs[5]** – `angular_velocity`：角速度
- **obs[6]** – `left_support_contact`：左腿/左支撑板是否接触（1.0 接触，0.0 未接触）
- **obs[7]** – `right_support_contact`：右腿/右支撑板是否接触（1.0 接触，0.0 未接触）

---

## 4. 动作空间 action_space
- **type**: Discrete(4)  
- **动作 0** – `no_engine`：不点火，无推力  
- **动作 1** – `left_orientation_engine`：点燃左侧姿态调整引擎（产生转向力）  
- **动作 2** – `main_engine`：点燃主引擎（产生主推力，一般向上）  
- **动作 3** – `right_orientation_engine`：点燃右侧姿态调整引擎（与 action 1 反向）

---

## 5. step 与终止条件分析
### 5.1 终止模式
终止触发由布尔表达式决定：  
`terminated = crash_or_body_contact OR horizontal_position_outside_viewport OR body_not_awake_or_settled`

- **success-like termination**：  
  - `body_not_awake_or_settled`：机身稳定、停止运动（通常因双腿着地且速度/角速度足够小）。  
  - `crash_or_body_contact` 在降落成功时也会为真（双腿接触平台），两种条件可能同时满足。  
  → 成功可由 `terminated=True` 时的状态特征判断（接近原点、小速度、双腿接触、小角度）。

- **failure-like termination**：  
  - `horizontal_position_outside_viewport`：水平漂出视野，必然失败。  
  - `crash_or_body_contact` 在**非安全接触**（如高速撞击地面、侧翻、头部触地）时触发，属于失败。  
  → 失败状态一般表现出大速度、大角度、位置偏移、单腿或零腿接触。

- **ambiguous termination**：  
  - 无显式成功/失败标志，所有终止都需要通过 `obs` 内容区分。

- **truncation**：  
  - 未定义时间截断，目前返回的 `truncated` 为 `False`。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: **false**  
- **explicit_failure_flag_available**: **false**  
- **allowed_info_fields**: 无（`step` 返回 `{}`，禁止使用 info）  
- **forbidden_or_uncertain_info_fields**: 所有 info 字段不可用；原始奖励 `original_reward` 被屏蔽，不可访问。

---

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

**允许使用**：
- `obs`（终止前一步的观察，或任意步骤的当前观察）
- `action`（当前执行的动作）
- `next_obs`（执行动作后的观察）
- `info` 仅限空字典，无可用字段
- `training_progress` 仅在明确被允许时使用，此处不开放

**禁止使用**：
- `original_reward`（官方奖励已屏蔽）
- 任何未声明的 info 字段
- 未在观察含义中声明的 obs 切片
- 外部环境属性或时间信息

---

## 7. 可用于奖励函数的信号
基于观察与动作，可构造的奖励驱动信号包括：

- **位置**：`next_obs[0]`（水平偏差），`next_obs[1]`（垂直偏差） → 衡量接近目标中心  
- **速度**：`next_obs[2]`（水平速度），`next_obs[3]`（垂直速度） → 控制软着陆  
- **朝向**：`next_obs[4]`（角度） → 保持竖直  
- **角速度**：`next_obs[5]`（角速度） → 抑制旋转  
- **接触**：`next_obs[6]`（左腿接触），`next_obs[7]`（右腿接触） → 奖励双腿平稳着地  
- **动作/引擎使用**：`action` ID → 惩罚燃料消耗（主引擎与姿态引擎的代价可区分）

---

## 8. 不确定或不可用的信号
- **原始奖励**（`original_reward`）完全屏蔽，不可用。  
- **info 字典**为空，无任何终止原因、成功标志或接触类型信息。  
- **是否坠毁 / 安全着陆** 无直接 flag，必须用观察（速度、角度、位置、接触）间接推断，且可能存在边界模糊情况。  
- **任务时间 / 步数** 未暴露，不能直接用于奖励中的时间惩罚。
