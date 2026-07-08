# Response Record

# Env_001 环境理解卡片

## 1. 任务目标
在一个 2D 物理环境中，控制一个带有主引擎和两个姿态引擎的飞行器。飞行器初始位置在视口顶部中央附近，受到随机初始力影响。核心目标是从初始位置出发，**尽可能快地降落到画面中央的目标平台上**，并稳定地停在平台上（速度趋于零、姿态水平、支撑腿安全接触）。次要要求是在完成主要目标的前提下，**尽量减少引擎推力使用量**（节省燃料）。简而言之：到达中央平台、快速稳定、省燃料。

## 2. 任务类型选择
selected_route_id: navigation_goal_reaching

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: 通常为 float32
- obs[0]: 飞行器相对于目标平台的水平坐标 x（目标平台中心 x=0）
- obs[1]: 飞行器相对于目标平台高度的垂直坐标 y（平台表面高度为 0）
- obs[2]: 水平线速度 vx
- obs[3]: 垂直线速度 vy
- obs[4]: 飞行器身体朝向角（弧度），0 表示水平
- obs[5]: 角速度（弧度/秒）
- obs[6]: 左侧支撑腿接触标志（0.0：未接触，1.0：接触）
- obs[7]: 右侧支撑腿接触标志（0.0：未接触，1.0：接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine（不点火，仅受重力/惯性影响）
- action 1: left_orientation_engine（点燃左侧姿态引擎，产生向右的力矩或侧向推力）
- action 2: main_engine（点燃主引擎，产生向上的推力）
- action 3: right_orientation_engine（点燃右侧姿态引擎，产生向左的力矩或侧向推力）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `body_not_awake_or_settled`（身体静止/稳定）。当飞行器停止运动（可能已着陆且满足稳定条件）时触发，很可能代表成功降落并稳定。但需结合位置、接触等信息进一步确认。
- failure-like termination: `crash_or_body_contact`（碰撞地面或不当身体接触），`horizontal_position_outside_viewport`（水平飞出可视区域）。这两种情况通常意味着撞击地面、偏离目标区域等失败结局。
- ambiguous termination: `body_not_awake_or_settled` 也可能在失败后（如侧翻卡住）触发，不能绝对等同于成功。
- truncation: 环境步始终返回 truncated=False，无时间限制信号。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {} （info 字典为空，无可用的成功/失败标记字段）
- forbidden_or_uncertain_info_fields: info 中不提供任何字段；终止条件的具体判定逻辑不透明，不能作为奖励计算依据；禁止使用任何模拟器内部信号。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs`：当前观测，可索引全部 8 个字段
- `action`：执行的动作（0~3 整数）
- `next_obs`：下一时刻观测，同样可索引全部字段
- `info`：空字典，无任何可用字段
- `training_progress`：未在任务描述中授权使用，因此**禁止使用**

禁止使用：
- `original_reward`（被遮蔽的官方奖励）
- 任何基于终止条件（如 crash、outside、settled）的硬编码成功/失败信号
- 任何未经声明的 info 字段
- 模拟器内部状态或额外全局信息

## 7. 可用于奖励函数的信号
- position: `obs[0]`（x 相对目标），`obs[1]`（y 相对平台高度）；`next_obs[0]`, `next_obs[1]`
- velocity: `obs[2]`（vx），`obs[3]`（vy）；以及下一时刻的对应值
- orientation: `obs[4]`（角度），`obs[5]`（角速度）
- contact: `obs[6]`（左腿接触），`obs[7]`（右腿接触），可用于判断着陆状态
- action/engine: 动作 ID 可以反映引擎使用情况（主引擎、姿态引擎、无操作），可用于惩罚燃料消耗
- 可构建的组合特征：与目标的距离、速度大小、是否水平、是否双触地、摆动幅度等

## 8. 不确定或不可用的信号
- 明确的成功/失败标记：info 为空，终止原因无法直接获取，不能用于奖励
- 官方原始奖励：`original_reward` 被遮蔽，严禁使用
- 终止条件的内部实现：`body_not_awake_or_settled`、`crash_or_body_contact` 等判断逻辑未知，不能作为奖励的直接输入
- 任何需要直接访问物理引擎或真实任务名的信息
