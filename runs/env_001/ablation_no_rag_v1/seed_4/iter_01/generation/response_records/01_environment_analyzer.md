# Response Record

# 匿名环境理解卡片

## 1. 任务目标
一个二维飞行器从视口顶部中央附近出发（带有随机初始扰动），需要尽可能快地到达并稳定停靠在中央目标平台上，同时尽量减少发动机推力。智能体应当学会靠近目标、减速、保持稳定姿态并安全接触平台。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box  
- shape: [8]  
- dtype: float32（推断）  
- obs[0]: 相对于目标平台的水平坐标  
- obs[1]: 相对于平台高度的垂直坐标  
- obs[2]: 水平线速度  
- obs[3]: 垂直线速度  
- obs[4]: 机身朝向角度  
- obs[5]: 角速度  
- obs[6]: 左侧支撑接触标志（0.0或1.0）  
- obs[7]: 右侧支撑接触标志（0.0或1.0）

## 4. 动作空间 action_space
- type: Discrete (n=4)  
- action 0: no_engine —— 不启动任何引擎  
- action 1: left_orientation_engine —— 启动左姿态调节引擎（产生逆时针/右转力矩）  
- action 2: main_engine —— 启动主引擎（提供向上推力）  
- action 3: right_orientation_engine —— 启动右姿态调节引擎（产生顺时针/左转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（机身静止或已稳定，推测代表成功着陆并稳定）  
- failure-like termination: crash_or_body_contact（碰撞/危险身体接触）、horizontal_position_outside_viewport（水平位置超出边界）  
- ambiguous termination: 无（但body_not_awake_or_settled是否为绝对成功需结合任务理解，通常视为成功）  
- truncation: 未在 step 源码中提供（不排除有最大步数限制，但未暴露，不可假设）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
- explicit_failure_flag_available: false  
- allowed_info_fields: 无（info 固定为空字典 {}）  
- forbidden_or_uncertain_info_fields: 所有 info 字段（因为 info 为空，任何假设都不安全）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（完整观察）
- action（当前执行的动作）
- next_obs（下一观察，如果需要可提取差值）
- info 中明确允许的字段（当前无任何允许字段）

禁止使用：
- original_reward（被屏蔽，不可依赖）
- official_reward 或任何类似原始奖励
- 未在观察空间中声明的 obs 切片（但这里 obs 已经全部声明，无隐藏字段；但仍不可使用未知索引）
- info 中的任何字段（info 固定为空，不允许假设）
- training_progress（该参数未在任务描述中被明确允许，禁止使用）

## 7. 可用于奖励函数的信号
- position: obs[0]（相对水平位置）、obs[1]（相对高度）  
- velocity: obs[2]（水平速度）、obs[3]（垂直速度）  
- orientation: obs[4]（角度）、obs[5]（角速度）  
- contact: obs[6]（左接触）、obs[7]（右接触）  
- action/engine: 通过 action 参数获知当前使用的引擎类型（是否开主引擎、姿态引擎等）

## 8. 不确定或不可用的信号
- 明确的成功/失败标志（info 为空，termination 条件由环境内部判断，reward 中无法直接获取）  
- 是否发生了安全接触（仅靠 contact 标志可推断接触，但无法区分安全着陆与碰撞）  
- 训练进度/步数限制（training_progress 不被允许，且无其他事件计数）  
- 任何原始奖励或官方奖励（被屏蔽）  
- 任何未在上述观察范围内的传感器读数（如燃料量、质量等，均未提供）
