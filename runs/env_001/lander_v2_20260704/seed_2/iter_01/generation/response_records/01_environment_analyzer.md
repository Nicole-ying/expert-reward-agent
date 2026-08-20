# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 飞行器轨迹优化任务。智能体从靠近视口顶部中心的位置出发，并承受一个随机初始作用力。目标是**尽快飞行并稳定降落在中央目标平台上，同时尽可能节省发动机推力**。智能体需要学习：平滑接近目标区域、控制下落速度、保持机身姿态稳定，并安全地让两侧支撑腿同时接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box  
- shape: [8,]  
- dtype: 通常为 float32（环境未显式声明，按连续控制环境惯例）  
- 各维度含义（index 从 0 开始）：
  - `obs[0]`: **x_position** —— 机体相对于目标平台中心的水平距离（相对坐标）  
  - `obs[1]`: **y_position** —— 机体相对于目标平台支撑面高度的垂直距离（相对坐标）  
  - `obs[2]`: **x_velocity** —— 机体水平线速度  
  - `obs[3]`: **y_velocity** —— 机体垂直线速度  
  - `obs[4]`: **body_angle** —— 机身倾斜角度（朝向）  
  - `obs[5]`: **angular_velocity** —— 机身绕质心的角速度  
  - `obs[6]`: **left_support_contact** —— 左侧支撑腿是否接触（0.0 或 1.0）  
  - `obs[7]`: **right_support_contact** —— 右侧支撑腿是否接触（0.0 或 1.0）

## 4. 动作空间 action_space
- type: Discrete(4)  
- 离散动作含义：
  - `action 0`: **no_engine** —— 不开启任何发动机，被动受重力与惯性影响  
  - `action 1`: **left_orientation_engine** —— 点燃左侧姿态控制发动机（产生旋转力矩并伴随微小推力）  
  - `action 2`: **main_engine** —— 点燃主发动机（产生向上的主推力，同时可能伴随偏航力矩）  
  - `action 3`: **right_orientation_engine** —— 点燃右侧姿态控制发动机（与 action 1 方向相反的姿态控制）

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**：`body_not_awake_or_settled` 当机身静止/稳定时触发。若此时机身位于目标平台区域、姿态接近水平且两腿均接触，则该终止可能对应成功着陆。但由于步骤函数未提供成功标记，此终止本质上仍具有歧义性。
- **failure-like termination**：  
  - `crash_or_body_contact` —— 机身本体（非支撑腿）撞击地面或其它障碍物，几乎一定意味着失败。  
  - `horizontal_position_outside_viewport` —— 水平位置超出允许边界，通常为飞出任务区域，属于失败。  
- **ambiguous termination**：  
  - `body_not_awake_or_settled` 若在平台外触发（如悬空稳定、在边界外停住或翻倒后静止），则属于失败；但因缺乏显式的成功/失败旗标，无法在终止时直接分辨。  
- **truncation**：代码片段未展示步数限制，但许多类似环境有 `max_steps` 截断；当前信息不足，暂不列入可用信号。

### 5.2 success/failure 信号可用性
- `explicit_success_flag_available`: **false**  
- `explicit_failure_flag_available`: **false**  
- `allowed_info_fields`: `[]`（`info` 始终返回空字典 `{}`，未提供任何额外字段，亦无 `success`/`failure` 等字段）  
- `forbidden_or_uncertain_info_fields`: 所有 `info` 字段均不可用（因信息为空，强行使用会导致 KeyError 或不可预料行为）。特别说明：`info["success"]`、`info["failure"]`、`info["termination_reason"]` 等均不存在，严禁假设。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：上一时刻的观察数组（8维）  
- `action`：当前步骤执行的动作（0/1/2/3）  
- `next_obs`：执行动作后的观察数组（8维）  
- `info` 中明确允许的字段（当前为空，即不可用）  
- `training_progress`：**仅当 prompt 中显式说明可以使用时才可访问**；此处无说明，故应视为禁止使用

严格禁止使用：
- `original_reward`（被遮盖的官方奖励，严禁读取或拟合）  
- 任何未声明的 `info` 字段  
- 任何未在 3 中声明含义的 `obs` / `next_obs` 索引切片（如超出 0-7 的索引）  
- 环境内部状态或通过其他手段获得的 ground truth (如绝对位置、目标平台坐标等)

## 7. 可用于奖励函数的信号
- **位置**：`obs[0]`（水平相对距离）、`obs[1]`（垂直相对距离）及 `next_obs` 对应值，可构建距离惩罚或接近奖励  
- **速度**：`obs[2]`、`obs[3]`，用于惩罚过大的线速度（软着陆要求）  
- **朝向/角速度**：`obs[4]`、`obs[5]`，用于惩罚倾斜姿态或旋转  
- **接触**：`obs[6]`、`obs[7]`，两腿接触状态可用于奖励安全着陆（两腿同时接触）或惩罚单腿/无接触  
- **动作/发动机使用**：`action` 本身可用于计算节省燃料的代价（如动作=2 主发动机时惩罚，动作=0 时奖励）  
- 上述信号均可在 `compute_reward` 中通过 `obs`、`action`、`next_obs` 安全获取。

## 8. 不确定或不可用的信号
- 环境在终止时**不提供任何显式的成功/失败标记**，因此无法从 `info` 或特殊字段直接判断任务是否完成。  
- 原始官方奖励 `original_reward` 已被遮蔽，不可用。  
- `info` 中不存在任何字段，故无法使用 `info` 进行奖励塑形。  
- 无法获取绝对目标坐标或绝对视口边界，只能依赖相对位置 `obs[0:1]`。  
- 由于无显式成功标记，终止条件 `body_not_awake_or_settled` 自身不能直接作为奖励信号（需要结合位置/接触判断）。
