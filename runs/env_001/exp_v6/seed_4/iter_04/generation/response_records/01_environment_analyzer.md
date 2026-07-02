# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
本环境为一个 2D 飞行器轨迹优化任务。智能体初始位于视口顶部中央附近，受到随机初始力。目标是尽快到达中央目标垫（着陆平台）并稳定着陆，同时尽可能节省引擎推力。智能体需要学会接近目标、减小速度、保持稳定姿态并安全接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching  
confidence: high  
reason: 任务核心是引导飞行器从初始位置到达并稳定在目标点（中央目标垫），典型的导航目标到达问题；同时存在引擎燃料节省、姿态稳定等辅助目标，但仍以位置收敛为主。

## 3. 观察空间 observation_space
- type: Box (连续向量)
- shape: (8,)
- dtype: float64
- obs[0]: x_position —— 相对于目标垫的水平坐标（越接近 0 越对准目标）
- obs[1]: y_position —— 相对于着陆平台顶面的垂直坐标（越接近 0 越贴近平台）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体倾斜角度（弧度）
- obs[5]: angular_velocity —— 机体角速度
- obs[6]: left_support_contact —— 左支撑脚接触标志（1.0 接触，0.0 未接触）
- obs[7]: right_support_contact —— 右支撑脚接触标志（1.0 接触，0.0 未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: 无操作（no_engine）—— 不启用任何引擎，仅凭惯性飞行
- action 1: 左姿态引擎（left_orientation_engine）—— 启动一个方向姿态调整引擎
- action 2: 主引擎（main_engine）—— 启动主引擎产生推力（通常向上减速）
- action 3: 右姿态引擎（right_orientation_engine）—— 启动相反方向姿态调整引擎

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  当满足 body_not_awake_or_settled（机体静止/休眠）且可能同时位于目标垫上方/接触时，视为成功稳定着陆。但终止条件中**没有明确判定是否在目标垫上**，仅凭“静止”无法保证成功，需要结合位置和接触信息推测。
- failure-like termination:  
  - crash_or_body_contact —— 机体发生非期望碰撞（如与地面、非目标物体接触）  
  - horizontal_position_outside_viewport —— 水平位置越界（掉出视口）
- ambiguous termination:  
  body_not_awake_or_settled 若发生在非目标位置或未接触平台，可能属于失败（如悬停后坠落或停在边缘），但终止条件本身混合了成功与失败语义。
- truncation:  
  仅当达到最大步数时由环境外部截断（本环境 step 返回 `False` 给 truncation，说明不会主动截断，但可能由上层框架限制步数）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []  （info 字典为空，无任何可用字段）
- forbidden_or_uncertain_info_fields: 任何 info 字段均不可用，尤其是不可依赖 `info["success"]`、`info["failure"]`、`info["termination_reason"]` 等常见但未提供的键。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs —— 当前观察向量（8维）
- action —— 当前执行的动作（0-3）
- next_obs —— 下一时刻观察向量
- info 中明确允许的字段（当前无允许字段）
- training_progress —— 仅在 prompt 明确要求使用时可用，本环境卡片不强制使用

禁止使用：
- original_reward —— 被屏蔽的原始奖励，严禁直接或间接拷贝
- official_reward —— 同上
- 未声明的 info 字段
- 未声明的 obs 切片解释（如自行猜测其他物理量）

## 7. 可用于奖励函数的信号
- position: obs[0] (相对目标水平距离), obs[1] (相对平台高度) —— 可用于引导靠近目标垫
- velocity: obs[2] (水平速度), obs[3] (垂直速度) —— 可用于鼓励减速、平稳着陆
- orientation: obs[4] (机体倾斜角) —— 可用于鼓励保持垂直姿态
- angular_velocity: obs[5] —— 可用于惩罚剧烈旋转
- contact: obs[6] (左脚接触), obs[7] (右脚接触) —— 可用于判断着陆成功并给予正奖励
- action/engine: action == 2（主引擎）可用于惩罚燃料消耗；action == 1 或 3 可用于惩罚过度姿态调整

## 8. 不确定或不可用的信号
- 原始奖励（被屏蔽）
- 任何显式成功/失败标志（info 为空）
- 目标垫接触以外的碰撞类型（crash_or_body_contact 无法细分为“致命碰撞”和“安全接触”，因终止条件未给出详细分类）
- 剩余步数或时间信息
