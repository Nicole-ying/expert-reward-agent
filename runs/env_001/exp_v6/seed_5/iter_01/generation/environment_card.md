# Env_001 环境理解卡片

## 1. 任务目标
这是一个 2D 类车辆轨迹优化任务。智能体（类似着陆器）从视口顶部中心附近以随机初速度开始，目标是尽快到达并稳定停留在中心目标垫上，同时尽量少使用引擎推力。期望行为是：接近目标、降低速度、保持姿态稳定并实现安全接触。

## 2. 任务类型选择
selected_route_id: multi_objective_task
confidence: high
reason: 任务需要同时优化“尽快到达”（时间最小化）和“尽可能少的引擎推力”（燃料经济性），这两个目标可能冲突。此外还需要稳定姿态和安全接触，整体构成典型的多目标控制问题。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32
- obs[0]: x_position – 水平方向相对于目标垫的坐标
- obs[1]: y_position – 垂直方向相对于垫面高度的坐标
- obs[2]: x_velocity – 水平线速度
- obs[3]: y_velocity – 垂直线速度
- obs[4]: body_angle – 机体朝向角度
- obs[5]: angular_velocity – 角速度
- obs[6]: left_support_contact – 左支撑接触标志（1.0=接触，0.0=未接触）
- obs[7]: right_support_contact – 右支撑接触标志（1.0=接触，0.0=未接触）

## 4. 动作空间 action_space
- type: Discrete(4)
- action 0: no_engine – 无任何引擎推力
- action 1: left_orientation_engine – 启动左姿态引擎（提供旋转力矩）
- action 2: main_engine – 启动主引擎（向上推力，减少下落速度）
- action 3: right_orientation_engine – 启动右姿态引擎（提供反向旋转力矩）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
  - body_not_awake_or_settled：机体不再活跃或已稳定，若此时满足垫上接触且姿态/速度足够小，可视为成功着陆（但环境未提供显式标志）。
- failure-like termination:
  - crash_or_body_contact：发生坠毁或错误接触（如碰撞、硬着陆）
  - horizontal_position_outside_viewport：水平位置超出视口范围
- ambiguous termination:
  - body_not_awake_or_settled 同样可能因坠毁而不再活跃，需结合观测区分
- truncation: 无（环境未使用 truncated 信号，step 返回 truncated=False）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不存在

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（终止前状态）
- action（执行的动作）
- next_obs（执行动作后的状态）
- info（当前为空字典）
- training_progress（仅在 prompt 明确授权时使用，当前环境为非必要）

禁止使用：
- original_reward（官方奖励，已被屏蔽）
- 任何未在上述列表中声明的变量或 info 字段
- 官方奖励的任何估计或重构

## 7. 可用于奖励函数的信号
- position: obs[0] x_position（水平偏移）、obs[1] y_position（相对垫面高度）
- velocity: obs[2] x_velocity、obs[3] y_velocity
- orientation: obs[4] body_angle、obs[5] angular_velocity
- contact: obs[6] left_support_contact、obs[7] right_support_contact
- action/engine: action 索引（0~3）用于评估燃料使用或惩罚特定动作

## 8. 不确定或不可用的信号
- 官方成功/失败标志：info 无相关字段
- 环境内部的时间步计数或燃料消耗量（未暴露）
- 任何未在 obs 中直接给出的语义标签（如 “成功着陆”）
- 外界风力或随机扰动（仅描述提及，未量化）