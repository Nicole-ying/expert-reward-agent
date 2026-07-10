# Duplicate reward retry
The previous generation duplicated iter 1 (runs/env_002/paper_bipedal_v4/seed_0/iter_01/generation/reward_v1.py). Retry 2: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python

```

# Search objective
- target_score: 300.000000
- current_score: 300.224778
- gap_to_target: -0.224778
- target_achievement_ratio: 100.075%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 300.224778）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 前进速度奖励（只奖励正向移动）
    hor_vel = next_obs[2]
    forward_reward = 2.0 * max(0.0, hor_vel)

    # 2. 平衡/稳定惩罚：身体倾角与角速度的平方惩罚
    angle = next_obs[0]
    ang_vel = next_obs[1]
    stability_penalty = -2.0 * (angle ** 2) - 0.5 * (ang_vel ** 2)

    # 3. 节能惩罚：关节力矩平方和
    energy_penalty = -0.005 * (action[0] ** 2 + action[1] ** 2 +
                               action[2] ** 2 + action[3] ** 2)

    total_reward = forward_reward + stability_penalty + energy_penalty

    components = {
        "forward_velocity": forward_reward,
        "stability": stability_penalty,
        "energy": energy_penalty
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=300.224778, len=1078.050000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[299.070953, 301.485751]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_velocity | 1008.781804 | 95.1% | 95.1% | 99.3% |
| stability | -43.264310 | -4.1% | 4.1% | 100.0% |
| energy | -8.251407 | -0.8% | 0.8% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境要求一个平面双足身体在起伏不平的地形上持续前进。主目标是**尽可能远、尽可能快地向前移动**，同时**尽可能减少能量消耗**（动作幅度/力矩）。次要目标是**维持上身稳定、避免摔倒**，因为一旦身体倾覆或到达地形尽头，回合即终止。因此奖励设计需引导智能体学习高效、平稳的前进步态，而不是到达某个固定终点。

## 3. 观察空间 observation_space
- type: Box
- shape: [24]
- dtype: float32（推断）
- 各维度说明与 reward_usable 标记：

| 索引 | 名称 | 含义 | reward_usable |
|------|------|------|---------------|
| 0 | hull_angle | 主体相对于竖直方向的倾角（越大越倾斜） | true |
| 1 | hull_angular_velocity | 主体倾角的角速度 | true |
| 2 | horizontal_velocity | 前进/后退方向的线速度（正值=向前） | true |
| 3 | vertical_velocity | 上下方向线速度（正值=上升） | true |
| 4 | hip1_angle | 腿1髋关节角度 | true |
| 5 | hip1_speed | 腿1髋关节角速度 | true |
| 6 | knee1_angle | 腿1膝关节角度 | true |
| 7 | knee1_speed | 腿1膝关节角速度 | true |
| 8 | leg1_contact | 腿1是否触地（1.0=触地，0.0=悬空） | true |
| 9 | hip2_angle | 腿2髋关节角度 | true |
| 10 | hip2_speed | 腿2髋关节角速度 | true |
| 11 | knee2_angle | 腿2膝关节角度 | true |
| 12 | knee2_speed | 腿2膝关节角速度 | true |
| 13 | leg2_contact | 腿2是否触地 | true |
| 14–23 | lidar_0 … lidar_9 | 10个激光雷达测距值，描述前方地形轮廓 | true |

所有维度均可作为奖励信号来源，但雷达数据仅反映地形几何，直接用于奖励需要谨慎映射（如惩罚前方陡坡）。

## 4. 动作空间 action_space
- type: Box
- shape: [4]
- 连续，各维范围 [-1.0, 1.0]
- 动作维度含义：

| 索引 | 名称 | 含义 |
|------|------|------|
| 0 | hip_torque_leg1 | 施加在腿1髋关节上的力矩 |
| 1 | knee_torque_leg1 | 施加在腿1膝关节上的力矩 |
| 2 | hip_torque_leg2 | 施加在腿2髋关节上的力矩 |
| 3 | knee_torque_leg2 | 施加在腿2膝关节上的力矩 |

动作幅度的平方和可作为能量消耗的代理指标。

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination**：`reached_end_of_terrain` – 走完前方全部地形，通常代表任务圆满达成。
- **failure-like termination**：`body_fallen_over` – 身体倾覆，失去平衡，代表失败。
- **ambiguous termination**：默认 `terminated` 为布尔值，在 step 返回时无法区分上述两种原因。
- **truncation**：未提及超时截断，推测由环境内部最大步数控制，但在本次 `step` 接口中未见传递。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false  
  （`info` 字典为空，无法获取明确的成功标记）
- **explicit_failure_flag_available**: false  
  （同上）
- **allowed_info_fields**：无（info = {}）
- **forbidden_or_uncertain_info_fields**：全部 info 子字段均为不可用，不得依赖任何未声明的键。

**关键结论**：奖励函数**不可基于终止原因**来区分成功/失败（如不可给到达终点额外奖励），因为无法从 `terminated` 布尔值可靠分离两种终止情况。奖励逻辑必须全部基于状态‑动作历程的内部信号构建。

## 7. 可用于奖励函数的信号
- **位置相关**：雷达测距（lidar_*）提供前方地形相对距离，可间接反映位置与地形起伏，但不提供绝对位置（无x坐标）。
- **速度相关**：`horizontal_velocity`（前进速度，可直接奖励）、`vertical_velocity`（可用于检测跳跃、坠落或异常运动）。
- **姿态/平衡相关**：`hull_angle`、`hull_angular_velocity`，身体倾角与角速度。
- **接触相关**：`leg1_contact`、`leg2_contact`，可用于鼓励交替触地，或惩罚双腿同时离地。
- **动作/力矩相关**：`action` 各维（力矩），平方和或绝对值可作为能耗惩罚。
- **关节状态**：四条关节角度与角速度，可用于惩罚过大的关节运动或极限位置。
- **地形预览**：lidar_0 … lidar_9，可辅助判断是否为陡坡或平台，谨慎作为奖励调整项。

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | energy + forward_velocity + stability | 300.22 | 300.22 | 0.00 | 1078.05 | energy=-0.010 forward_velocity=0.556 stability=-0.036 | target_solved_new_best |
