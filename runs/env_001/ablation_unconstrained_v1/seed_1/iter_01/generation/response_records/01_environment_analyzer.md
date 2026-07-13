# Response Record

# 匿名环境理解卡片

## 1. 任务目标
这是一个 2D 类飞行器着陆/姿态控制任务。  
主体从屏幕上方中部某位置出发，受到初始随机作用力。  
目标是**尽快到达并稳定降落在中央目标平台**上，同时**尽可能少使用引擎推力**。  
智能体需要学习：  
- 向目标平台靠近  
- 降低速度  
- 保持稳定姿态  
- 与平台安全接触（两条支撑腿同时着地）  

附属优化包括省燃料、时间短、姿态平稳，但不额外构成独立目标。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (8,)
- dtype: float32 或 float64（具体未知，但连续数值）
- obs[0]: x_position —— 水平坐标，相对于目标平台（零点为目标中心）
- obs[1]: y_position —— 垂直坐标，相对于平台高度（零点为平台表面高度）
- obs[2]: x_velocity —— 水平线速度
- obs[3]: y_velocity —— 垂直线速度
- obs[4]: body_angle —— 机体朝向角度（可能用弧度）
- obs[5]: angular_velocity —— 角速度
- obs[6]: left_support_contact —— 左支撑腿接触标志，1.0 表示接触，0.0 表示不接触
- obs[7]: right_support_contact —— 右支撑腿接触标志，1.0 表示接触，0.0 表示不接触

## 4. 动作空间 action_space
- type: Discrete
- 动作数量: 4
- action 0: no_engine —— 不启动任何引擎，无推力/力矩
- action 1: left_orientation_engine —— 启动左姿态引擎（产生顺时针或逆时针力矩，具体方向需经验判断）
- action 2: main_engine —— 启动主引擎（产生向上的主要推力）
- action 3: right_orientation_engine —— 启动右姿态引擎（产生与动作1相反的力矩）

动作空间是针对飞行器的简单推力与姿态控制，无油门大小，每个动作固定输出一个脉冲。

## 5. step 与终止条件分析
### 5.1 终止模式
环境在满足以下任一条件时终止（terminated = True）：
- **crash_or_body_contact** —— 主体部分（非支撑腿）发生碰撞或接触（可能摔到地面上、撞到障碍等）
- **horizontal_position_outside_viewport** —— 水平坐标超出视口范围（远离目标平台过远）
- **body_not_awake_or_settled** —— 身体不再运动或已稳定（包括成功着陆后稳定静止）

> 可见，最后一种可能既包含成功着陆稳定，也可能包含其他静止的状态（如卡住、惯性停止等）。需要结合上下文判断。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false   （info 中没有 success 字段）
- explicit_failure_flag_available: false   （info 中没有 failure 字段）
- allowed_info_fields: （info 为空字典，无任何字段可用）
- forbidden_or_uncertain_info_fields: info 中所有字段不可用（因为 info 为空）；不可使用 terminated 标志

**注意**：终止条件（crash/出界/稳定）的真实语义不能直接从 `info` 获得，需要从 `next_obs` 的状态推理。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs —— 转换前的当前观测
- action —— 刚刚执行的动作（整数0~3）
- next_obs —— 转换后的观测（用于推断状态变化和终止原因）
- info 中明确允许的字段 —— 当前 info 为空，所有 info 字段都禁止使用
- training_progress —— **当前任务描述未授权使用，默认禁止使用**，除非 prompt 明确说明允许

禁止使用：
- original_reward 或任何官方奖励
- 未声明的 info 字段
- 未声明的 obs 切片（例如不能假设 obs[0] 存在之外的含义）
- 任何形式的终止标志（terminated, done）

奖励函数设计时，**不得直接访问环境内部的终止条件**，只能通过 `next_obs` 的状态特征间接推断成功或失败。

## 7. 可用于奖励函数的信号
- **位置（接近目标）**：`next_obs[0]`（相对目标平台的水平距离），`next_obs[1]`（高度偏移），可鼓励 x,y 趋向 0
- **速度（着陆柔和性）**：`next_obs[2]`, `next_obs[3]`，可惩罚过大的速度，尤其垂向着陆速度
- **姿态**：`next_obs[4]`（身体角度），鼓励接近 0（水平姿态）；`next_obs[5]`（角速度），惩罚剧烈旋转
- **接触**：`next_obs[6]` 和 `next_obs[7]`（左右支撑腿接触），两条腿同时接触标志稳定的成功着陆
- **动作/引擎使用**：从 `action` 可以判断是否使用了引擎（0为无引擎，其他为有引擎），可惩罚燃料消耗；也可以结合燃料效率同时使用 `obs` 与 `next_obs` 的变化判断推力作用

## 8. 不确定或不可用的信号
- **成功/失败标签**：不存在于 info，不能直接获取
- **终止原因分类**：无法得知是 crash、出界还是稳定，只能从 next_obs 中的位置、速度、接触状态进行概率性推断
- **距离目标的确收指标**：没有明确的“着陆成功”布尔信号，需要通过位置足够近、速度接近零、双腿接触等组合条件来近似
- **训练进度信息**：默认不允许，除非将来 prompt 明确授权
- **任何隐藏的内部状态**：例如风力、摩擦力、随机初始冲量等，都不在观测中

---

**总结**：这个环境是一个典型的导航/着陆任务（navigation_goal_reaching），奖励需要从位置、速度、姿态、接触状态和动作中手工构造，以引导快速且省燃料地平稳着陆到平台中心。
