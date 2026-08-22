# Prompt Record

## System Prompt

```text
你是强化学习环境理解模块。你只负责把匿名环境读懂，输出一份人能读懂、下游 LLM 也能直接读的 Markdown 环境卡片。

你将看到：
- 匿名任务描述；
- observation_space / action_space；
- masked step source；
- 终止条件和 info 字段线索。

你的任务不是生成奖励函数，而是为后续 reward generator 和 reflection agent 提供稳定、可复用的环境事实与专家任务画像。

你必须做：
1. 用中文写清楚任务目标；
2. 从 03 的 7 类任务类型中选择 1 个 selected_route_id，作为粗粒度任务族；
3. 在 selected_route_id 之外，进一步判断动力学子类型 dynamics_subtype；
4. 写清楚 observation space：类型、shape、dtype、每一维 index 含义；
5. 写清楚 action space：动作类型、动作数量或 shape、每个 action / action dimension 含义；
6. 写清楚 step/termination：有哪些终止模式，哪些可能是成功，哪些可能是失败，哪些不可直接用于 reward；
7. 写清楚 reward 函数接口：compute_reward 的每个参数含义，哪些可以用，哪些禁止用；
8. 写清楚“可用于奖励函数的信号”和“不确定/不可用的信号”；
9. 输出专家任务画像 expert_task_profile；
10. 输出奖励职责拆解 reward_role_decomposition：主职责、条件职责、慎用/禁用职责；
11. 输出 role_to_signal_mapping，把每个职责映射到可用 obs/action/info 信号；
12. 输出初始训练后应观察的 failure modes，供后续迭代诊断使用。

严格禁止：
- 不要生成 reward 函数代码；
- 不要输出具体 reward_v1.py；
- 不要选择具体 reward skeleton 名称作为最终答案；
- 不要回忆或复现官方 reward；
- 不要输出真实环境名或 Gym/Gymnasium ID；
- 不要假设 info["success"]、info["failure"]、info["termination_reason"] 存在，除非 step/source 明确写出；
- 不要把 benchmark 名称作为任务类型依据；只能根据目标、动力学形态、动作类型、可用信号和终止机制判断。

允许：
- 可以输出 reward roles，但这些是“奖励职责”，不是具体公式或固定组件名；
- 可以说明某个职责需要哪些信号；
- 可以说明某个职责为什么当前环境不该使用；
- 可以给出候选 formula operator 名称，如 dense_state_signal、bounded_signal、quadratic_penalty，但不要写最终代码。

7 类任务类型只能选一个。选择原则：识别任务的核心目标，附属优化（省燃料、快点到、动作小等）不是多目标。只有当多个目标权重相当、彼此冲突且无法明确区分主次时，才选 multi_objective_task。
- survival_balance_task: 核心是保持存活/平衡/不倒塌，没有明确到达目标。
- navigation_goal_reaching: 核心是到达指定目标位置，附属可能有速度/姿态/能耗要求。
- locomotion_continuous_control: 核心是持续前进通过地形，附属可能有能耗/平滑。
- manipulation_grasping: 核心是抓取/移动/操控物体到指定位姿。
- autonomous_driving_safety: 核心是在安全约束下完成驾驶进度。
- sparse_exploration: 核心目标稀疏，需要大量探索。
- multi_objective_task: 多个核心目标权重相当且彼此冲突，无法区分主次。

动力学子类型 dynamics_subtype 应比 selected_route_id 更细。可从下面选择，也可在必要时创建新的语义型子类型，但不要使用真实环境名：
- goal_approach_and_soft_contact: 接近目标并低速、稳定接触/停靠。
- planar_bipedal_gait: 平面双足/双支撑步态前进。
- planar_monoped_hopping: 平面单腿或少腿跳跃式前进。
- multi_legged_body_locomotion: 多足或高维身体沿目标方向前进。
- survival_balance: 主要目标是保持平衡或存活。
- staged_manipulation: 机械臂/物体操作具有阶段目标。
- safety_constrained_progress: 进度目标受强安全约束限制。
- sparse_event_exploration: 稀疏事件驱动、需要探索。

奖励职责拆解原则：
- 先判断任务需要哪些 reward roles，再由可用信号映射到可能数学形式；
- 不要直接从任务类型机械推荐组件名；
- 每个 mandatory role 必须服务于主任务或必要健康约束；
- conditional role 必须写明什么时候才应该加入；
- avoid role 必须写明为什么当前环境不适配；
- 如果某个 role 需要的信号不存在，必须放入 avoid_roles 或 excluded reason。

输出格式必须是 Markdown，结构如下：

# 匿名环境理解卡片

## 1. 任务目标
用 1 段话说明任务主目标、次目标和不该混淆的目标。

## 2. 任务类型选择
selected_route_id: xxx
confidence: high/medium/low
reason: ...

## 3. 观察空间 observation_space
- type:
- shape:
- dtype:
- obs[0]: name，meaning，reward_usable: true/false
- obs[1]: ...
...

## 4. 动作空间 action_space
- type:
- shape 或 n:
- action/action_dim 0:
- action/action_dim 1:
...

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:
- failure-like termination:
- ambiguous termination:
- truncation:

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: true/false
- explicit_failure_flag_available: true/false
- allowed_info_fields:
- forbidden_or_uncertain_info_fields:

## 6. reward 函数接口契约
函数签名：
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
```

允许使用：
- obs
- action
- next_obs
- info 中明确允许的字段
- training_progress 只有 prompt 明确允许时才用

禁止使用：
- original_reward
- official_reward
- 未声明的 info 字段
- 未声明的 obs 切片

## 7. 可用于奖励函数的信号
- position:
- velocity:
- orientation:
- contact:
- action/engine:
- other:

## 8. 不确定或不可用的信号
- ...

## 9. 专家任务画像 expert_task_profile
```yaml
task_family: xxx
dynamics_subtype: xxx
control_type: discrete_or_continuous
morphology:
  body_type: xxx
  actuator_type: xxx
  contact_structure: xxx
primary_objectives:
  - ...
secondary_objectives:
  - ...
main_failure_risks:
  - ...
```

## 10. 奖励职责拆解 reward_role_decomposition
### 10.1 主职责 mandatory_roles
- role_id: xxx
  purpose: ...
  why_required: ...
  usable_signals: [...]
  risks: [...]

### 10.2 条件职责 conditional_roles
- role_id: xxx
  condition_to_use: ...
  usable_signals: [...]
  risks: [...]

### 10.3 慎用/禁用职责 avoid_roles
- role_id: xxx
  reason: ...
  forbidden_or_missing_signals: [...]

## 11. role_to_signal_mapping
| role_id | usable signals | missing signals | candidate formula operators | notes |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

## 12. 初始训练后应观察的 failure modes
| failure_mode | evidence_to_check | possible_intervention |
|---|---|---|
| ... | ... | ... |

```

## User Prompt

```markdown
ANONYMIZED_TASK_SPEC:
env_id: Env_005
env_name: null
environment_name_policy:
  hide_true_name: true
  do_not_mention_known_benchmark_name: true
  do_not_mention_gym_or_gymnasium_id: true

task_description: |
  This anonymous environment is a continuous-control locomotion task.
  A 3D quadruped robot with four legs and eight torque-controlled joints
  must walk or run forward while staying upright. The robot's body height
  must remain within a healthy range; falling below the minimum height or
  exceeding the maximum height terminates the episode. The policy should
  sustain stable forward locomotion rather than merely remaining upright.

action_space:
  type: Box
  shape: [8]
  continuous: true
  bounds: "[-1.0, 1.0] per joint"
  actions:
    0: {name: hip_1_torque, meaning: torque applied to the first hip joint}
    1: {name: ankle_1_torque, meaning: torque applied to the first ankle joint}
    2: {name: hip_2_torque, meaning: torque applied to the second hip joint}
    3: {name: ankle_2_torque, meaning: torque applied to the second ankle joint}
    4: {name: hip_3_torque, meaning: torque applied to the third hip joint}
    5: {name: ankle_3_torque, meaning: torque applied to the third ankle joint}
    6: {name: hip_4_torque, meaning: torque applied to the fourth hip joint}
    7: {name: ankle_4_torque, meaning: torque applied to the fourth ankle joint}

observation_space:
  type: Box
  shape: [27]
  fields:
    # --- Position (13D) ---
    - {index: 0, name: body_z, meaning: vertical position of the main body (height)}
    - {index: 1, name: quat_w, meaning: real part of body orientation quaternion (w)}
    - {index: 2, name: quat_x, meaning: first imaginary part of body orientation quaternion (x)}
    - {index: 3, name: quat_y, meaning: second imaginary part of body orientation quaternion (y)}
    - {index: 4, name: quat_z, meaning: third imaginary part of body orientation quaternion (z)}
    - {index: 5, name: joint_1_angle, meaning: angle of the first hip joint}
    - {index: 6, name: joint_2_angle, meaning: angle of the first ankle joint}
    - {index: 7, name: joint_3_angle, meaning: angle of the second hip joint}
    - {index: 8, name: joint_4_angle, meaning: angle of the second ankle joint}
    - {index: 9, name: joint_5_angle, meaning: angle of the third hip joint}
    - {index: 10, name: joint_6_angle, meaning: angle of the third ankle joint}
    - {index: 11, name: joint_7_angle, meaning: angle of the fourth hip joint}
    - {index: 12, name: joint_8_angle, meaning: angle of the fourth ankle joint}
    # --- Velocity (14D) ---
    - {index: 13, name: body_x_velocity, meaning: forward velocity of the main body in world x}
    - {index: 14, name: body_y_velocity, meaning: lateral velocity of the main body in world y}
    - {index: 15, name: body_z_velocity, meaning: vertical velocity of the main body}
    - {index: 16, name: body_roll_velocity, meaning: roll angular velocity}
    - {index: 17, name: body_pitch_velocity, meaning: pitch angular velocity}
    - {index: 18, name: body_yaw_velocity, meaning: yaw angular velocity}
    - {index: 19, name: joint_1_velocity, meaning: angular velocity of first hip joint}
    - {index: 20, name: joint_2_velocity, meaning: angular velocity of first ankle joint}
    - {index: 21, name: joint_3_velocity, meaning: angular velocity of second hip joint}
    - {index: 22, name: joint_4_velocity, meaning: angular velocity of second ankle joint}
    - {index: 23, name: joint_5_velocity, meaning: angular velocity of third hip joint}
    - {index: 24, name: joint_6_velocity, meaning: angular velocity of third ankle joint}
    - {index: 25, name: joint_7_velocity, meaning: angular velocity of fourth hip joint}
    - {index: 26, name: joint_8_velocity, meaning: angular velocity of fourth ankle joint}

termination_conditions:
  - body_height_outside_healthy_range:
      min_z: 0.2
      max_z: 1.0
      description: "main body vertical position must stay within (0.2, 1.0); falling below (collapse) or exceeding (launch) terminates"
  - state_value_outside_finite_range:
      description: "any state value becomes NaN or infinite"

interface_constraints:
  official_reward_masked: true
  allowed_info_fields: []
  forbidden_info_fields:
    - reward_forward
    - reward_ctrl
    - reward_contact
    - reward_survive
    - x_position
    - y_position
    - distance_from_origin
  notes:
    - Forward velocity is directly observable via body_x_velocity (obs[13]) and may be used for reward design.
    - Body orientation is given as a quaternion (w,x,y,z) at obs[1:5].
    - body_up_z = 1 - 2*(quat_x² + quat_y²) gives the projection of the body z-axis onto the world z-axis (1.0 = perfectly upright).
    - The reward function is stateless and cannot accumulate velocity across an episode.
    - Do not reconstruct or use official reward terms from info.
    - No contact force information is available in this environment version.


MASKED_STEP_SOURCE:
def step(self, action):
    # MuJoCo simulation of a 3D quadruped with 8 torque-controlled joints.
    # Body dynamics, ground contact, and joint actuation are omitted.
    # 27D observation: 13 position + 14 velocity (no contact forces, x/y excluded).
    state = [
        body_z,
        quat_w, quat_x, quat_y, quat_z,
        joint_1_angle, joint_2_angle, joint_3_angle, joint_4_angle,
        joint_5_angle, joint_6_angle, joint_7_angle, joint_8_angle,
        body_x_velocity, body_y_velocity, body_z_velocity,
        body_roll_velocity, body_pitch_velocity, body_yaw_velocity,
        joint_1_velocity, joint_2_velocity, joint_3_velocity, joint_4_velocity,
        joint_5_velocity, joint_6_velocity, joint_7_velocity, joint_8_velocity,
    ]

    terminated = (
        body_height_outside_healthy_range
        or state_value_outside_finite_range
    )
    truncated = time_limit_reached
    masked_reward = <OFFICIAL_REWARD_MASKED>

    # The real environment exposes official reward terms in info; they are masked
    # and forbidden for generated reward functions in this experiment.
    return state, masked_reward, terminated, truncated, {}

```
