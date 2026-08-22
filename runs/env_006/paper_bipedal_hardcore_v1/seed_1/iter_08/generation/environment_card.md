# 匿名环境理解卡片

## 1. 任务目标
训练一个双足机器人（两条腿）在布满梯子、树桩、坑洞等不规则障碍的崎岖地形上尽可能远地向前行走，同时尽量节省能量。理想情况下，机器人应平稳走到地形尽头（到达终点），全程避免摔倒，并保持较低的关节力矩消耗。激光雷达提供前方地面高度测量，机器人需要利用这些信息提前调整步态以通过障碍。

## 2. 任务类型选择
selected_route_id: locomotion_continuous_control
confidence: high
reason: 核心任务是让机器人在粗糙地形上持续向前行走，前进距离是主要指标；到达终点可视为前进的自然结果，能量效率是次要优化目标，不存在多个同等权重且相互冲突的核心目标。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断）
- obs[0]: hull_angle，身体倾斜角，reward_usable: true
- obs[1]: hull_angular_velocity，身体角速度，reward_usable: true
- obs[2]: horizontal_speed，质心水平速度（前进方向），reward_usable: true
- obs[3]: vertical_speed，质心垂直速度，reward_usable: true
- obs[4]: joint_0_angle（hip_1），髋关节1角度，reward_usable: true
- obs[5]: joint_0_speed（hip_1），髋关节1角速度，reward_usable: true
- obs[6]: joint_1_angle（knee_1），膝关节1角度，reward_usable: true
- obs[7]: joint_1_speed（knee_1），膝关节1角速度，reward_usable: true
- obs[8]: joint_2_angle（hip_2），髋关节2角度，reward_usable: true
- obs[9]: joint_2_speed（hip_2），髋关节2角速度，reward_usable: true
- obs[10]: joint_3_angle（knee_2），膝关节2角度，reward_usable: true
- obs[11]: joint_3_speed（knee_2），膝关节2角速度，reward_usable: true
- obs[12]: leg_1_ground_contact，第一腿触地指示（1.0/0.0），reward_usable: true
- obs[13]: leg_2_ground_contact，第二腿触地指示（1.0/0.0），reward_usable: true
- obs[14]–obs[23]: lidar_1 到 lidar_10，共10个激光测距读数（前方地形高度），reward_usable: true

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- bounds: [-1.0, 1.0]
- action_dim 0: hip_1_torque，髋关节1扭矩
- action_dim 1: knee_1_torque，膝关节1扭矩
- action_dim 2: hip_2_torque，髋关节2扭矩
- action_dim 3: knee_2_torque，膝关节2扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain`（到达地形尽头，视为成功）
- failure-like termination: `body_fallen_over`（摔倒，视为失败）
- ambiguous termination: 无
- truncation: 无（episode 必定以终止而非超时结束）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: []（info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

终止原因无法从 info 直接读取，但可通过观测信号间接推断：
- 摔倒：可由 `hull_angle` 突然增大、`vertical_speed` 快速下降、或双腿离地等异常模式推断（derived_possible）
- 到达终点：可由 episode 突然终止且未检测到明显摔倒信号推断（derived_possible）

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs（当前步观测）
- action（当前步动作）
- next_obs（下一步观测，可基于差分生成增量信号）
- info 中明确允许的字段（本环境 info 为空，实际不可用）
- training_progress 仅在 prompt 明确允许时使用（此处未允许）

禁止使用：
- original_reward（严禁）
- official_reward（严禁）
- 任何未在观察空间或 info 中明确声明的信号
- 终止标志 terminated 作为奖励直接输入（即使可从函数外部传入，也应避免依赖，以免信息泄露）

## 7. 可用于奖励函数的信号
- position / velocity: `horizontal_speed`（前进速度），`vertical_speed`（垂直速度，可检测跳跃/坠落）
- orientation: `hull_angle`（倾斜角度），`hull_angular_velocity`（角速度）
- joint states: 4个关节角度和角速度，可用于平滑性、对称性、关节限位检查
- contact: `leg_1_ground_contact`、`leg_2_ground_contact`（触地标志）
- action / torque: `action[0..3]` 四个扭矩值
- LIDAR: 10个测距值，可提取前方地形信息，辅助判断步态适应性
- derived_possible: 从 `hull_angle`、`vertical_speed`、`contact` 等推断摔倒风险；从 episode 意外终止且无摔倒迹象推断到达终点（仅限诊断，不建议直接加入 reward，以免不稳定）

## 8. 不确定或不可用的信号
- 绝对位置坐标（x, y）不可用，无法直接计算前进距离
- 距离终点距离不可用
- 没有任何显式 success/failure 标志
- info 字段全部不可用
- 外部地形高度图不可用（仅有前方10个LIDAR点）

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: locomotion_continuous_control
dynamics_subtype: planar_bipedal_gait_on_rough_terrain
control_type: continuous (torque)
morphology:
  body_type: bipedal (2 hips, 2 knees)
  actuator_type: torque-controlled joints
  contact_structure: leg-ground contact (2 binary sensors)
  perception: LIDAR rangefinders (10 forward-facing)
primary_objectives:
  - 最大化前进距离（通过正水平速度推动）
  - 保持平衡，避免摔倒
secondary_objectives:
  - 最小化不必要的关节力矩消耗
main_failure_risks:
  - 倾倒（hull_angle 过大）
  - 被障碍物绊住而停滞
  - 高能耗导致早期耗尽（若考虑 accumulated cost）
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: forward_progress
  purpose: 鼓励机器人向前移动，最大化水平速度。
  why_required: 核心任务目标是“尽可能远地前进”，水平速度是前进的直接指标。
  usable_signals: [horizontal_speed, x‑velocity]
  risks: 可能诱导机器人在障碍前盲目加速而摔倒，需配合生存约束。

- role_id: survival_balance
  purpose: 惩罚可能导致摔倒的姿态与运动，保证机器人存活。
  why_required: 摔倒即终止且无效，必须作为强约束。
  usable_signals: [hull_angle, hull_angular_velocity, vertical_speed, leg_contact 模式]
  risks: 惩罚过强可能使机器人不敢移动；需与前进奖励平衡。

### 10.2 条件职责 conditional_roles
- role_id: efficient_actuation
  condition_to_use: 当机器人已具备稳定前进能力（平均水平速度超过某阈值）时启用，避免初期扼杀探索。
  usable_signals: [action 扭矩向量（平方和或绝对值和）]
  risks: 若过早启用会抑制必要的大力矩动作，导致无法学会起步或跨越障碍。

- role_id: gait_smoothness
  condition_to_use: 当机器人已有基本步态后，可奖励关节角速度的低方差，使步伐更平稳。
  usable_signals: [joint speeds, joint accelerations derived from next_obs - obs]
  risks: 可能降低对粗糙地形的适应能力，需仅作为微小 bonus。

### 10.3 慎用/禁用职责 avoid_roles
- role_id: goal_reaching_bonus
  reason: 缺少终点距离信号，无法直接激励“靠近终点”；若通过终止推断终点而给予一次性奖励，会引入信息泄露且极不稳定。
  forbidden_or_missing_signals: [distance_to_goal, explicit_reached_end]

- role_id: terrain_adaptation_explicit
  reason: 虽然LIDAR存在，但尚不明确如何设计“正确利用LIDAR”的奖励；直接基于LIDAR设计奖励极易导致策略利用传感器而非真正学习迈步，风险极高，建议初期避免。
  forbidden_or_missing_signals: 缺乏地形高度与步长、抬脚高度之间的明确映射。

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| forward_progress | horizontal_speed | absolute_position, distance_traveled | dense_state_signal (linear/quadratic of v_x) | 对负水平速度应给予强惩罚 |
| survival_balance | hull_angle, hull_angular_velocity, vertical_speed, leg_contact | explicit_fallen | bounded_signal (penalize when abs(angle) > threshold), quadratic_penalty | 可结合 contact 检测双离地异常 |
| efficient_actuation | action (4D torque vector) | – | quadratic_penalty (torque magnitude) | 作为条件组件，其权重需随训练进度逐渐增大或 gate by forward_progress |
| gait_smoothness | joint_speeds (obs[5,7,9,11]) | – | variance_penalty, bounded_signal | 注意避免与 terrain 适应性冲突；初期可关闭 |
| goal_reaching_bonus | derived_possible (terminated with no fall signs) | distance_to_goal | (禁用) | 不推荐用于 reward，可在分析中作为 episode 成功标志 |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| 频繁摔倒（hull_angle 大幅摆动） | hull_angle 经常超出安全范围，episode 很短 | 提高 survival_balance 权重，降低 forward_progress 权重；添加脚部接触一致性惩罚