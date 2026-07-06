# Response Record

# 匿名环境理解卡片

## 1. 任务目标
控制一个平面单腿关节体连续向前跳跃，同时保持身体直立、各关节在健康范围内。  
策略必须持续稳定地前进，而不仅仅是维持站立。一旦身体高度、躯干角度或任何状态值超出健康范围，回合即终止（失败）。  
任务期望通过关节扭矩输出实现高效、持续的跳跃前进。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control

## 3. 观察空间 observation_space
- type: Box
- shape: (11,)
- dtype: float32
- obs[0]: torso_height — 主体垂直高度
- obs[1]: torso_angle — 主体朝向角（弧度）
- obs[2]: upper_joint_angle — 上腿关节角度
- obs[3]: lower_joint_angle — 下腿关节角度
- obs[4]: foot_joint_angle — 脚关节角度
- obs[5]: forward_velocity — 主体水平速度（前向为正）
- obs[6]: vertical_velocity — 主体垂直速度
- obs[7]: torso_angular_velocity — 主体角速度
- obs[8]: upper_joint_speed — 上腿关节角速度
- obs[9]: lower_joint_speed — 下腿关节角速度
- obs[10]: foot_joint_speed — 脚关节角速度

## 4. 动作空间 action_space
- type: Box（连续）
- 形状: (3,)
- 范围: [-1.0, 1.0] 每关节扭矩
- action 0: upper_joint_torque — 上腿铰链扭矩
- action 1: lower_joint_torque — 下腿铰链扭矩
- action 2: foot_joint_torque — 脚铰链扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: 无显式成功终止，任务目标为持续前进，不设成功停止条件。
- failure-like termination: 
  - `body_height_outside_healthy_range`（躯干高度超出健康范围，如摔倒）
  - `torso_angle_outside_healthy_range`（躯干倾角过大，如倾倒）
  - `state_value_outside_finite_healthy_range`（任意状态值出现NaN或无穷）
- ambiguous termination: 无
- truncation: `time_limit_reached`（达到最大步数时截断，并非失败）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 字典为空）
- forbidden_or_uncertain_info_fields:
  - `reward_forward`
  - `reward_ctrl`
  - `reward_survive`
  - `x_position`
  - `y_position`
  - `z_distance_from_origin`
  （上述字段虽然可能存在于真实环境的 info 中，但已被屏蔽并禁止使用）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- `obs` （当前观察）
- `action` （执行的动作）
- `next_obs` （下一观察）
- `info` 中明确允许的字段（当前允许为空列表，因此 info 中无可用字段）
- `training_progress` **仅在 prompt 明确允许时使用**，本任务未明确允许，故应视为不可用

禁止使用：
- `original_reward`（已屏蔽）
- 任何 `official_reward` 或其变体
- 未声明的 info 字段（如 `x_position`, `reward_forward` 等）
- 对观察空间未声明的切片或隐藏状态

## 7. 可用于奖励函数的信号
- position: `torso_height`, `torso_angle`（垂直位置、姿态角；注意不含世界x坐标）
- velocity: `forward_velocity`（前向速度，可直接奖励前进）, `vertical_velocity`, `torso_angular_velocity`, 各关节角速度
- orientation: `torso_angle` （衡量直立）
- contact: 无直接接触信号
- action/engine: `upper_joint_torque`, `lower_joint_torque`, `foot_joint_torque`（用于惩罚能耗或大幅动作）

## 8. 不确定或不可用的信号
- 世界前向位移 `x_position`：被禁止，不可从观察或 info 获得
- 其它世界坐标 `y_position`, `z_distance_from_origin`：同样不可用
- 官方分解奖励项（`reward_forward`, `reward_ctrl`, `reward_survive`）：禁用
- 身体与地面的接触力、足端触地状态：观察中无此类信号
- 外部累积量（如总前向距离）：奖励函数为无状态，单步内不能累计历史
