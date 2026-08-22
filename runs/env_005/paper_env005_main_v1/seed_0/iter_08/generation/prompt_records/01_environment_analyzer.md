# Prompt Record

## System Prompt

```text
你是强化学习环境理解模块。你只负责把匿名环境读懂，输出一份人能读懂、下游 LLM 也能直接读的 Markdown 环境卡片。

你将看到：
- 匿名任务描述；
- observation_space / action_space；
- masked step source；
- 终止条件和 info 字段线索。

你必须做：
1. 用中文写清楚任务目标；
2. 从 03 的 7 类任务类型中选择 1 个 selected_route_id；
3. 写清楚 observation space：类型、shape、dtype、每一维 index 含义；
4. 写清楚 action space：动作类型、动作数量、每个 action id 含义；
5. 写清楚 step/termination：有哪些终止模式，哪些可能是成功，哪些可能是失败，哪些不可直接用于 reward；
6. 写清楚 reward 函数接口：compute_reward 的每个参数含义，哪些可以用，哪些禁止用；
7. 写清楚“可用于奖励函数的信号”和“不确定/不可用的信号”。

严格禁止：
- 不要设计奖励函数；
- 不要选择 reward skeleton；
- 不要输出 v1_candidate_skeletons；
- 不要回忆官方 reward；
- 不要输出真实环境名或 Gym/Gymnasium ID；
- 不要假设 info["success"]、info["failure"]、info["termination_reason"] 存在，除非 step/source 明确写出。

7 类任务类型只能选一个。选择原则：识别任务的核心目标，附属优化（省燃料、快点到、动作小等）不是多目标。——只有当多个目标权重相当、彼此冲突且无法明确区分主次时，才选 multi_objective_task。
- survival_balance_task: 核心是保持存活/平衡/不倒塌，没有明确到达目标（如倒立摆、平衡杆）
- navigation_goal_reaching: 核心是到达指定目标位置，附属可能有速度/姿态/能耗要求（如着陆、导航）
- locomotion_continuous_control: 核心是持续前进通过地形，附属可能有能耗/平滑（如步行、驾驶）
- manipulation_grasping: 核心是抓取/移动/操控物体到指定位姿
- autonomous_driving_safety: 核心是在安全约束下完成驾驶进度（如车道保持、避障）
- sparse_exploration: 核心目标是稀疏的，需要大量探索（如迷宫、稀疏奖励导航）
- multi_objective_task: 多个核心目标权重相当且彼此冲突，无法区分主次

输出格式必须是 Markdown，结构如下：

# 匿名环境理解卡片

## 1. 任务目标
...

## 2. 任务类型选择
selected_route_id: xxx   （此行不要用 Markdown 加粗格式，纯文本即可）

## 3. 观察空间 observation_space
- type:
- shape:
- dtype:
- obs[0]:
- obs[1]:
...

## 4. 动作空间 action_space
- type:
- action 0:
- action 1:
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

## 8. 不确定或不可用的信号
- ...

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
  shape: [105]
  part_structure:
    - {part: joint_positions, indices: [0,12], size: 13, meaning: "angular positions of all joints and body orientation (world x,y excluded)"}
    - {part: joint_velocities, indices: [13,26], size: 14, meaning: "angular and linear velocities of all joints and body"}
    - {part: contact_forces, indices: [27,104], size: 78, meaning: "external contact forces and torques on each body part (6D per part, 13 parts, excluding world body)"}
  fields:
    - {index: 0, name: body_z, meaning: vertical position of the main body}
    - {index: 1, name: body_orientation_1, meaning: first component of body orientation quaternion}
    - {index: 2, name: body_orientation_2, meaning: second component of body orientation quaternion}
    - {index: 3, name: body_orientation_3, meaning: third component of body orientation quaternion}
    - {index: 4, name: body_orientation_4, meaning: fourth component of body orientation quaternion}
    - {index: 5, name: joint_1_angle, meaning: angle of the first leg joint}
    - {index: 6, name: joint_2_angle, meaning: angle of the second leg joint}
    - {index: 7, name: joint_3_angle, meaning: angle of the third leg joint}
    - {index: 8, name: joint_4_angle, meaning: angle of the fourth leg joint}
    - {index: 9, name: joint_5_angle, meaning: angle of the fifth leg joint}
    - {index: 10, name: joint_6_angle, meaning: angle of the sixth leg joint}
    - {index: 11, name: joint_7_angle, meaning: angle of the seventh leg joint}
    - {index: 12, name: joint_8_angle, meaning: angle of the eighth leg joint}
    - {index: 13, name: body_x_velocity, meaning: forward velocity of the main body in world x}
    - {index: 14, name: body_y_velocity, meaning: lateral velocity of the main body in world y}
    - {index: 15, name: body_z_velocity, meaning: vertical velocity of the main body}
    - {index: 16, name: body_roll_velocity, meaning: roll angular velocity}
    - {index: 17, name: body_pitch_velocity, meaning: pitch angular velocity}
    - {index: 18, name: body_yaw_velocity, meaning: yaw angular velocity}
    - {index: 19, name: joint_1_velocity, meaning: angular velocity of first leg joint}
    - {index: 20, name: joint_2_velocity, meaning: angular velocity of second leg joint}
    - {index: 21, name: joint_3_velocity, meaning: angular velocity of third leg joint}
    - {index: 22, name: joint_4_velocity, meaning: angular velocity of fourth leg joint}
    - {index: 23, name: joint_5_velocity, meaning: angular velocity of fifth leg joint}
    - {index: 24, name: joint_6_velocity, meaning: angular velocity of sixth leg joint}
    - {index: 25, name: joint_7_velocity, meaning: angular velocity of seventh leg joint}
    - {index: 26, name: joint_8_velocity, meaning: angular velocity of eighth leg joint}
    - {index: 27, name: contact_forces_start, meaning: "start of 78D contact force/torque block; see part_structure for layout"}
    # 78D contact forces = 13 body parts (torso + 12 leg segments) × 6D each
    # Each body part block: [fx, fy, fz, tx, ty, tz] in world coordinates
    # obs[27:33]   = torso contact
    # obs[33:39]   = front_left upper leg contact
    # obs[39:45]   = front_left lower leg contact
    # obs[45:51]   = front_left foot contact       ← high ground-contact signal for this leg
    # obs[51:57]   = front_right upper leg contact
    # obs[57:63]   = front_right lower leg contact
    # obs[63:69]   = front_right foot contact      ← high ground-contact signal for this leg
    # obs[69:75]   = back_left upper leg contact
    # obs[75:81]   = back_left lower leg contact
    # obs[81:87]   = back_left foot contact        ← high ground-contact signal for this leg
    # obs[87:93]   = back_right upper leg contact
    # obs[93:99]   = back_right lower leg contact
    # obs[99:105]  = back_right foot contact       ← high ground-contact signal for this leg

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
    - x_velocity
    - y_velocity
  notes:
    - Forward velocity is directly observable via body_x_velocity (obs[13]) and may be used for reward design.
    - Contact forces in obs[27:104] provide per-leg ground contact information.
    - The reward function is stateless and cannot accumulate velocity across an episode.
    - Do not reconstruct or use official reward terms from info.


MASKED_STEP_SOURCE:
def step(self, action):
    # MuJoCo simulation of a 3D quadruped with 8 torque-controlled joints.
    # Body dynamics, ground contact, and joint actuation are omitted.
    state = [
        body_z,
        body_orientation_1, body_orientation_2, body_orientation_3, body_orientation_4,
        joint_1_angle, joint_2_angle, joint_3_angle, joint_4_angle,
        joint_5_angle, joint_6_angle, joint_7_angle, joint_8_angle,
        body_x_velocity, body_y_velocity, body_z_velocity,
        body_roll_velocity, body_pitch_velocity, body_yaw_velocity,
        joint_1_velocity, joint_2_velocity, joint_3_velocity, joint_4_velocity,
        joint_5_velocity, joint_6_velocity, joint_7_velocity, joint_8_velocity,
        # 78D external contact forces/torques on each body part
        contact_force_1_x, contact_force_1_y, contact_force_1_z,
        contact_torque_1_x, contact_torque_1_y, contact_torque_1_z,
        # ... (13 parts × 6D each = 78D)
        contact_force_13_x, contact_force_13_y, contact_force_13_z,
        contact_torque_13_x, contact_torque_13_y, contact_torque_13_z,
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
