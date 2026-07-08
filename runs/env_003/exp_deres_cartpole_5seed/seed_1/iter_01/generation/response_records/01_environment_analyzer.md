# Response Record

# Env_003 环境理解卡片

## 1. 任务目标
通过向移动底座施加固定大小的水平推力（负方向或正方向），使底座上的自由旋转杆保持竖直，同时底座自身不能越出轨道边界。目标是尽可能长时间地存活，即在底座位置超出 ±2.4 或杆倾角超出 ±0.2094 rad 前，持续控制。

## 2. 任务类型选择
selected_route_id: survival_balance_task
confidence: high
reason: 任务要求一直保持杆平衡和底座在轨道内，没有明确的到达目标，唯一的成功方式是尽可能长地避免终止，属于经典的生存‑平衡控制任务。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: [4]
- dtype: float32
- obs[0]: base_position — 移动底座的水平位置，范围 [-4.8, 4.8]（注意实际终止边界为 |base_position|>2.4）
- obs[1]: base_velocity — 底座水平速度，无界
- obs[2]: pole_angle — 杆相对于竖直方向的角度（rad），范围 [-0.41888, 0.41888]（终止边界为 |pole_angle|>0.20944）
- obs[3]: pole_angular_velocity — 杆的角速度，无界

## 4. 动作空间 action_space
- type: Discrete (2)
- action 0: push_negative_direction — 朝向轨道负方向施加固定推力
- action 1: push_positive_direction — 朝向轨道正方向施加固定推力

## 5. step 与终止条件分析
### 5.1 终止模式
- success‑like termination: 无
- failure‑like termination:
  - |base_position| > 2.4 （底座越界）
  - |pole_angle| > 0.20943951 rad （杆倾倒）
  任一条件满足即终止，可视为失败。
- ambiguous termination: 无
- truncation:
  - 达到 500 步自动截断（可视为存活成功）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: {}（info 为空，无可用字段）
- forbidden_or_uncertain_info_fields: 所有 info 键均不可用

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs (上一步观察)
- next_obs (当前步观察)
- action
- info 中明确允许的字段（当前无）

禁止使用：
- original_reward（官方奖励已隐藏，不可用）
- 任何 info 中的字段（info 为空）
- 基于环境内部状态的未声明信号

## 7. 可用于奖励函数的信号
- position: base_position (obs[0] / next_obs[0])
- velocity: base_velocity (obs[1] / next_obs[1])，pole_angular_velocity (obs[3] / next_obs[3])
- orientation: pole_angle (obs[2] / next_obs[2])
- action: 可获取当前选择的离散动作（0 或 1），用于动作平滑或惩罚高频切换等
- 隐式终止标记：通过检查 next_obs[0] 或 next_obs[2] 是否超出终止阈值（2.4 和 0.20944）来生成稀疏失败惩罚

## 8. 不确定或不可用的信号
- 原始环境奖励：已被遮蔽，不可用
- info 字典：为空，不包含任何额外信号（如 success、failure、终止原因等）
- 杆的绝对角度或位置之外的衍生状态：除非从已有观测中计算，否则不可用
