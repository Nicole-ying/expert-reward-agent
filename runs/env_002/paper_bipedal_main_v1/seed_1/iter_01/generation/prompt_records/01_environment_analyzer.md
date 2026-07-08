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
env_id: Env_002
env_name: null
environment_name_policy:
  hide_true_name: true
  do_not_mention_known_benchmark_name: true
  do_not_mention_gym_or_gymnasium_id: true

task_description: |
  This anonymous environment is a 2D locomotion task.
  A two-legged body must walk forward across uneven terrain as far and as fast as possible
  while minimizing energy consumption. The agent must coordinate hip and knee joints on
  both legs to produce a stable bipedal gait. Falling over terminates the episode.

action_space:
  type: Box
  shape: [4]
  continuous: true
  bounds: "[-1.0, 1.0] per joint"
  actions:
    0: {name: hip_torque_leg1, meaning: torque applied to hip joint of leg 1}
    1: {name: knee_torque_leg1, meaning: torque applied to knee joint of leg 1}
    2: {name: hip_torque_leg2, meaning: torque applied to hip joint of leg 2}
    3: {name: knee_torque_leg2, meaning: torque applied to knee joint of leg 2}

observation_space:
  type: Box
  shape: [24]
  fields:
    - {index: 0, name: hull_angle, meaning: angle of the main body relative to upright}
    - {index: 1, name: hull_angular_velocity, meaning: angular velocity of the main body}
    - {index: 2, name: horizontal_velocity, meaning: forward/backward linear velocity}
    - {index: 3, name: vertical_velocity, meaning: up/down linear velocity}
    - {index: 4, name: hip1_angle, meaning: hip joint angle of leg 1}
    - {index: 5, name: hip1_speed, meaning: hip joint angular speed of leg 1}
    - {index: 6, name: knee1_angle, meaning: knee joint angle of leg 1}
    - {index: 7, name: knee1_speed, meaning: knee joint angular speed of leg 1}
    - {index: 8, name: leg1_contact, meaning: ground contact flag for leg 1 (1.0=contact, 0.0=no contact)}
    - {index: 9, name: hip2_angle, meaning: hip joint angle of leg 2}
    - {index: 10, name: hip2_speed, meaning: hip joint angular speed of leg 2}
    - {index: 11, name: knee2_angle, meaning: knee joint angle of leg 2}
    - {index: 12, name: knee2_speed, meaning: knee joint angular speed of leg 2}
    - {index: 13, name: leg2_contact, meaning: ground contact flag for leg 2 (1.0=contact, 0.0=no contact)}
    - {index: 14..23, name: lidar_0..9, meaning: 10 LIDAR rangefinder distance measurements along the terrain ahead}

termination_conditions:
  - body_fallen_over
  - reached_end_of_terrain


MASKED_STEP_SOURCE:
def step(self, action):
    # Physics simulation, joint motor control, terrain contact and friction are omitted for compactness.
    # Official reward computation is masked and must not be reconstructed.
    state = [
        hull_angle,
        hull_angular_velocity,
        horizontal_velocity,
        vertical_velocity,
        hip1_angle,
        hip1_speed,
        knee1_angle,
        knee1_speed,
        1.0 if leg1_ground_contact else 0.0,
        hip2_angle,
        hip2_speed,
        knee2_angle,
        knee2_speed,
        1.0 if leg2_ground_contact else 0.0,
        lidar_0, lidar_1, lidar_2, lidar_3, lidar_4,
        lidar_5, lidar_6, lidar_7, lidar_8, lidar_9,
    ]
    terminated = body_fallen_over or reached_end_of_terrain
    masked_reward = <OFFICIAL_REWARD_MASKED>
    return state, masked_reward, terminated, False, {}

```
