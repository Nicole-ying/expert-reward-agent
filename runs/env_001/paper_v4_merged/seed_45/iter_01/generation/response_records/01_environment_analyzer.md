# Response Record

# 匿名环境理解卡片

## 1. 任务目标
本任务是一个 2D 载体轨迹优化问题。主体从一个随机初始位置（上部中央附近）开始，受随机初始推力影响。主要目标是使主体到达并稳定停靠在中央目标平台上，同时尽可能少地使用发动机推力（省燃料）。主体需要学会：精确接近目标、降低线速度与角速度、保持姿态稳定、安全接触平台。次要目标是快速完成和省燃料，但不能与安全着陆冲突。

## 2. 任务类型选择
selected_route_id: **navigation_goal_reaching**  
confidence: high  
reason: 任务核心是到达指定目标位置（中心目标垫）并稳定停靠，附属要求快和省燃料，这些是附属优化，不构成多目标冲突，因此属于导航目标到达族。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- 字段含义：
  - obs[0] `x_position`：相对目标垫中心的水平坐标，单位未指定，奖励可用 true
  - obs[1] `y_position`：相对目标垫高度的垂直坐标（正向可能代表高于垫），单位未指定，奖励可用 true
  - obs[2] `x_velocity`：水平线速度，奖励可用 true
  - obs[3] `y_velocity`：垂直线速度，奖励可用 true
  - obs[4] `body_angle`：主体朝向角（弧度，0为直立），奖励可用 true
  - obs[5] `angular_velocity`：角速度，奖励可用 true
  - obs[6] `left_support_contact`：左支撑脚接触标志（1.0=接触，0.0=未接触），奖励可用 true
  - obs[7] `right_support_contact`：右支撑脚接触标志（1.0=接触，0.0=未接触），奖励可用 true

所有维度均可直接或间接用于奖励函数。

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- 动作含义：
  - 0：`no_engine` — 不点火任何发动机，无推力
  - 1：`left_orientation_engine` — 点燃左侧姿态发动机，产生向左水平推力及/或旋转力矩（具体推力方向可影响水平速度与姿态角）
  - 2：`main_engine` — 点燃主发动机，产生垂直向上推力（对抗重力），同时可能产生微小力矩
  - 3：`right_orientation_engine` — 点燃右侧姿态发动机，产生向右水平推力及/或旋转力矩

注意：动作空间未描述精确力矩，但结合`body_angle`和`angular_velocity`，左右发动机可能同时影响水平加速度和角加速度。

## 5. step 与终止条件分析
### 5.1 终止模式
- **crash_or_body_contact**：主体非支撑部分撞击地面或与平台碰撞过猛导致坠毁（如角速度/速度过大）
- **horizontal_position_outside_viewport**：水平坐标超出视口范围（视为出界失败）
- **body_not_awake_or_settled**：主体进入“静止”或“稳定着陆”状态（可能包含成功着陆或长期静止）——这可能是成功着陆的主要终止触发器

没有显式的成功或失败标志。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false**
- explicit_failure_flag_available: **false**
- allowed_info_fields: {}（终端返回空的info字典）
- forbidden_or_uncertain_info_fields: 所有info字段均不可用。推断成功/失败只能通过观测信号组合与终止事件进行（derived_possible）：
  - 推断成功：终止时 `left_support_contact == 1 and right_support_contact == 1`，同时 `|x_position|` 和 `|y_position|` 接近0，`|x_velocity|`、`|y_velocity|`、`|body_angle|`、`|angular_velocity|` 均低于较小阈值。
  - 推断失败：终止时上述条件不满足，例如水平出界、或仅单脚接触、或角度/速度过大等。

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```
允许使用：
- obs
- action
- next_obs
- info 中无字段可用
- training_progress **禁止使用**（无明确许可）

禁止使用：
- original_reward（原始奖励已屏蔽）
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- **position**：x_position, y_position（相对目标垫中心坐标，可直接计算到目标(0,0)的距离）
- **velocity**：x_velocity, y_velocity
- **orientation**：body_angle, angular_velocity
- **contact**：left_support_contact, right_support_contact
- **action/engine**：当前 action（可用于燃料消耗惩罚，但无法知道推力大小，只能视为开关）
- **其他**：可从 next_obs 与 obs 构造差值（如 delta 位置、速度变化、角度变化），推断稳定性。

## 8. 不确定或不可用的信号
- 绝对高度或地面距离（仅知 y_position 相对垫高度，但垫高度未知，无法区分“很高”与“略高”，但可假设垫表面在 y=0 附近）
- 燃料剩余量（无fuel字段）
- 精确的坠落/碰撞信号（仅有终止事件，无 crash flag）
- 目标垫坐标（相对坐标隐含目标，可知目标在 (0,0)）
- 奖励累计、时间步长（无内部计时，仅 episode 长度）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: navigation_goal_reaching
dynamics_subtype: goal_approach_and_soft_contact
control_type: discrete
morphology:
  body_type: simplified_lunar_lander（两足着陆器）
  actuator_type: main_engine, left_side_thruster, right_side_thruster
  contact_structure: two_legs_with_contact_sensors
primary_objectives:
  - 到达目标平台（x≈0, y≈0）
  - 稳定着陆（两腿接触，低速，小倾角）
secondary_objectives:
  - 尽快完成（时间效率）
  - 最小化发动机使用（省燃料）
main_failure_risks:
  - 水平飞出视口
  - 过快的速度或大角度导致坠毁
  - 单脚着陆或翻滚
  - 在半空悬停消耗过多燃料
```

## 10. 奖励职责拆解 reward_role_decomposition

### 骨架选择推理（内部，不输出）
主问题：“离目标更近了吗？” → 选择 delta(distance) 主信号算子族。因为目标坐标已知，可用相对位置计算距离，并衡量两步之间改进。避免使用纯 proximity（如 -distance）作为唯一主信号，以防止 agent 悬停。

辅助信号：1) 终端成功/失败奖励（推断成功时大正，失败时大负）；2) 姿态稳定惩罚（角度、角速度超过安全阈值时增加 hinge 惩罚）；3) 燃料消耗惩罚（根据 action 是否为 1/2/3 给予微小负奖励，因为任务要求省燃料）；4) 着陆接触奖励（当双腿均接触时给予额外奖励，鼓励最终稳定）。

自检：如果 agent 静止不动（自由落体），delta distance 会根据下降和水平漂移变化：直落时 y 减小，距离减小，会得到正奖励，这合理，因为向目标接近；主信号会持续给正直到接触。agent 可能悬停（频繁点火），此时距离减少慢，但 delta 仍可给正；由于有燃料惩罚，可抑制过度悬停。完成着陆后 episode 终止，不会再收割奖励。

### 10.1 主职责 mandatory_roles
- **role_id: progress_toward_target**
  - purpose: 奖励每一步使 agent 更
