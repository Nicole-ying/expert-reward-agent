# 上一轮奖励函数代码（该轮得分: 270.713770）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 权重（可根据实际训练表现调整）
    w_forward = 1.0
    w_angle = 0.5
    w_vertical_vel = 0.1
    w_angular_vel = 0.05

    # 主学习信号：前进速度奖励
    forward_vel = next_obs[2]  # 水平速度，正值表示向前
    forward_reward = w_forward * forward_vel

    # 稳定/安全约束：抑制身体倾斜、跳动和高速旋转
    angle = abs(next_obs[0])           # 身体相对于竖直的角度
    vertical_vel = abs(next_obs[3])    # 上下跳动速度
    angular_vel = abs(next_obs[1])     # 身体转动角速度
    stability_penalty = - (w_angle * angle + w_vertical_vel * vertical_vel + w_angular_vel * angular_vel)

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
score=270.713770, len=1110.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[268.225794, 273.841199]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.877612 | 88.8% | 88.8% | 100.0% |
| stability_penalty | -63.436040 | -11.2% | 11.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
- 控制一个双足身体在不平坦的 2D 地形上向前行走。
- 核心要求：走得尽可能远、尽可能快，同时尽量降低能量消耗。
- 智能体需要协调两条腿的髋、膝关节，产生稳定的双足步态。
- 若身体摔倒，回合立即终止；若成功走到地形末端，回合也会终止。

## 3. 观察空间 observation_space
- type: Box (连续浮点数)
- shape: (24,)
- dtype: float32
- 观测向量各索引含义如下：
  - obs[0]: hull_angle —— 身体（躯干）相对于竖直方向的角度
  - obs[1]: hull_angular_velocity —— 身体的角速度
  - obs[2]: horizontal_velocity —— 前向/后向线速度
  - obs[3]: vertical_velocity —— 上下线速度
  - obs[4]: hip1_angle —— 腿1髋关节角度
  - obs[5]: hip1_speed —— 腿1髋关节角速度
  - obs[6]: knee1_angle —— 腿1膝关节角度
  - obs[7]: knee1_speed —— 腿1膝关节角速度
  - obs[8]: leg1_contact —— 腿1是否接触地面（1.0 = 接触，0.0 = 未接触）
  - obs[9]: hip2_angle —— 腿2髋关节角度
  - obs[10]: hip2_speed —— 腿2髋关节角速度
  - obs[11]: knee2_angle —— 腿2膝关节角度
  - obs[12]: knee2_speed —— 腿2膝关节角速度
  - obs[13]: leg2_contact —— 腿2是否接触地面（1.0 = 接触，0.0 = 未接触）
  - obs[14..23]: lidar_0..9 —— 10 个激光雷达测距值，返回前方地形距离

## 4. 动作空间 action_space
- type: Box (连续控制量)，范围 [-1.0, 1.0] 各关节独立
- shape: (4,)
- 动作索引与含义：
  - action 0: hip_torque_leg1 —— 施加在腿1髋关节的力矩
  - action 1: knee_torque_leg1 —— 施加在腿1膝关节的力矩
  - action 2: hip_torque_leg2 —— 施加在腿2髋关节的力矩
  - action 3: knee_torque_leg2 —— 施加在腿2膝关节的力矩

## 5. step 与终止条件分析

### 5.1 终止模式
- success-like termination: 到达地形末端（reached_end_of_terrain）—— 可能意味着成功走完全程，但没有显式的成功标志。
- failure-like termination: 身体摔倒（body_fallen_over）—— 典型失败终止。
- ambiguous termination: 无。
- truncation: 未提供任何时间步上限截断信息（返回的 truncated 均为 False，info 为空），任务可能默认依赖环境内部最大步数，但本文档无法确认。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无 success 字段）
- explicit_failure_flag_available: false （info 为空，无 failure 字段）
- allowed_info_fields: 无（info 始终为 {}，没有任何可用字段）
- forbidden_or_uncertain_info_fields:
  - info 的所有字段均不可用（因 info 为空）
  - 无法通过 info 获取 reached_end_of_terrain 或 body_fallen_over 的布尔标志（只能在 terminated 信号中推断，但 terminated 本身不区分成功/失败）
  - 不得假设任何额外的终止原因字符串（如 "termination_reason"）

## 7. 可用于奖励函数的信号
直接从 `obs` 或 `next_obs` 中可提取的物理含义信号：

- **身体姿态/稳定性**：
  - `hull_angle`（obs[0]）：身体倾斜角度，可用于惩罚偏离直立
  - `hull_angular_velocity`（obs[1]）：身体转动速度

- **线速度**：
  - `horizontal_velocity`（obs[2]）：前进速度，是衡量“走得快”的核心信号
  - `vertical_velocity`（obs[3]）：上下跳动速度

- **关节状态**：
  - 髋关节角度/速度：obs[4], obs[5], obs[9], obs[10]
  - 膝关节角度/速度：obs[6], obs[7], obs[11], obs[12]

- **地面接触**：
  - `leg1_contact`, `leg2_contact`（obs[8], obs[13]）：可用来鼓励双脚交替支撑、惩罚拖地等

- **动作（关节力矩）**：
  - action[0..3] 的绝对值或平方值，可作为能耗惩罚的直接信号

- **激光雷达（地形感知）**：
  - lidar_0..9（obs[14..23]）：前方地形距离信息，可用于评估地形难度或前方是否有障碍

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + stability_penalty | 270.71 | 270.71 | 0.00 | 1110.35 | forward_reward=0.355 stability_penalty=-0.044 | new_best |
