# Search objective
- target_score: 300.000000
- current_score: 103.027232
- gap_to_target: 196.972768
- target_achievement_ratio: 34.342%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 103.027232）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 前进速度奖励（正值表示向前，直接作为主学习信号）
    forward_speed = next_obs[2]
    forward_reward = forward_speed  # 系数 1.0

    # 稳定性惩罚：惩罚大倾角、高角速度和明显垂直速度（弹跳）
    tilt_angle = next_obs[0]
    angular_vel = next_obs[1]
    vertical_vel = next_obs[3]

    tilt_penalty = -0.5 * (tilt_angle ** 2)
    angular_vel_penalty = -0.1 * (angular_vel ** 2)
    vertical_vel_penalty = -0.5 * (vertical_vel ** 2)
    stability_penalty = tilt_penalty + angular_vel_penalty + vertical_vel_penalty

    total_reward = forward_reward + stability_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=103.027232, len=787.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-80.357059, 287.319037]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 319.959350 | 94.0% | 94.4% | 100.0% |
| stability_penalty | -19.094838 | -5.6% | 5.6% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个两足机器人（两条腿，每条腿有髋关节和膝关节）在不平坦的 2D 地形上稳定行走。  
主要目标是尽可能远、尽可能快地向前移动；次要目标是尽量降低能量消耗（动作力度）。  
机器人需要协调双腿的关节力矩，生成稳定的步态。一旦身体倒下，回合立即结束。

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: float32（假设，实际未见明确定义，通常连续观测为 float）  
- 各维度含义：  
  - obs[0]: hull_angle —— 身体（主躯干）相对于竖直方向的倾斜角  
  - obs[1]: hull_angular_velocity —— 身体角速度  
  - obs[2]: horizontal_velocity —— 前向/后向水平线速度  
  - obs[3]: vertical_velocity —— 上下线速度  
  - obs[4]: hip1_angle —— 腿 1 髋关节角度  
  - obs[5]: hip1_speed —— 腿 1 髋关节角速度  
  - obs[6]: knee1_angle —— 腿 1 膝关节角度  
  - obs[7]: knee1_speed —— 腿 1 膝关节角速度  
  - obs[8]: leg1_contact —— 腿 1 接地标志（1.0=接触地面，0.0=未接触）  
  - obs[9]: hip2_angle —— 腿 2 髋关节角度  
  - obs[10]: hip2_speed —— 腿 2 髋关节角速度  
  - obs[11]: knee2_angle —— 腿 2 膝关节角度  
  - obs[12]: knee2_speed —— 腿 2 膝关节角速度  
  - obs[13]: leg2_contact —— 腿 2 接地标志（1.0=接触地面，0.0=未接触）  
  - obs[14] ~ obs[23]: lidar_0 ~ lidar_9 —— 10 个前方地形激光雷达测距值（前方不同角度的距离）

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续动作，每个分量范围 [-1.0, 1.0]
- action 0: hip_torque_leg1 —— 施加于腿 1 髋关节的力矩（扭矩）
- action 1: knee_torque_leg1 —— 施加于腿 1 膝关节的力矩
- action 2: hip_torque_leg2 —— 施加于腿 2 髋关节的力矩
- action 3: knee_torque_leg2 —— 施加于腿 2 膝关节的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: `reached_end_of_terrain` —— 机器人到达地形的终点（任务目标完成）
- failure-like termination: `body_fallen_over` —— 身体倒下（任务失败）
- ambiguous termination: 无（从 step 源码看，terminated 只由这两个条件构成，不存在其他模糊情况）
- truncation: step 返回的第四个元素为 `False`，表示不存在时间截断，所有终止都是由环境状态触发的真实终止

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false  
  （`info` 字典为空，没有显式成功标志；只能从终止时是否为 `reached_end_of_terrain` 推断）
- explicit_failure_flag_available: false  
  （同上，需要根据是否为 `body_fallen_over` 推断）
- allowed_info_fields: 无（`info` 为空，下游 reward 函数不允许依赖任何 info 字段）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用，也不可假设存在官方 success/failure 标记

## 7. 可用于奖励函数的信号
从当前观测 `obs`、动作 `action` 和下一步观测 `next_obs` 中可以直接提取以下信号（需自行计算）：

- **前进速度**  
  - `obs[2]` / `next_obs[2]`：水平速度（正值前进，负值后退），可直接鼓励向前移动。
- **身体姿态稳定性**  
  - `obs[0]` / `next_obs[0]`：身体倾角，理想应接近 0（直立）；可惩罚大倾角。
  - `obs[1]`：身体角速度，鼓励小的角速度变化。
- **能量消耗（动作幅度）**  
  - `action` 的平方和或绝对值之和：反映当前控制力矩的大小，可用来惩罚过大动作以降低能耗。
- **关节状态与步态**  
  - `obs[4] ~ obs[7]`（腿 1 髋、膝角度/速度）、`obs[9] ~ obs[12]`（腿 2 对应值）：可设计周期性、摆动/支撑相位奖励。
  - `obs[8]` / `obs[13]`：接地标志，可用于检测步态相位（支撑相、摆动相），鼓励交替接地。
- **地形感知（激光雷达）**  
  - `obs[14:24]`：前方地形距离信息，可用于惩罚与障碍物过近的行为，但通常不作为主要前进奖励的直接来源。
- **终止时的结果推断**  
  - 虽然 `info` 为空，但可在训练循环中结合 `terminated` 标志与 `next_obs` 的状态判断：若水平速度接近零且身体倾角超过某阈值，可视为倒下；若 `terminated` 且身体直立、水平位置到达终点附近（需从环境外部获得位置信息，但此处观测中无位置，故无法直接获得终点信息），因此纯从观测难以区分成功终止与失败终止。设计奖励时需注意不要依赖终点识别。

# Compact expert route context
- selected_route_id: locomotion_continuous_control
- recommended_design_roles: forward_progress_reward (前进/速度奖励), terminal_failure_penalty (失败惩罚), energy_penalty (动作能耗惩罚), alive_bonus (存活奖励), action_smoothness_penalty (动作平滑惩罚), stability_penalty (姿态/稳定性惩罚)
- usage: These are design primitives and risk reminders, not mandatory components or a closed menu. Use only roles supported by current behavior evidence and declared inputs.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + stability_penalty | 103.03 | 103.03 | 0.00 | 787.75 | forward_reward=0.338 stability_penalty=-0.022 | new_best |
