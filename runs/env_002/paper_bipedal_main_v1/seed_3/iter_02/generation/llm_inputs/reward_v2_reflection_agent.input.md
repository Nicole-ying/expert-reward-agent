# 上一轮奖励函数代码（该轮得分: 289.987325）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 核心进步信号：鼓励沿前进方向快速移动
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # 姿态稳定约束：惩罚躯干倾角偏离直立
    hull_angle = next_obs[0]
    w_angle = 1.0  # 平方惩罚，对较大倾角敏感
    angle_penalty = -w_angle * (hull_angle ** 2)

    # 垂直稳定约束：抑制跳跃或剧烈起伏
    vertical_velocity = next_obs[3]
    w_vert = 0.5
    vert_penalty = -w_vert * (vertical_velocity ** 2)

    total_reward = forward_reward + angle_penalty + vert_penalty

    components = {
        'forward_reward': forward_reward,
        'angle_penalty': angle_penalty,
        'vert_penalty': vert_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=289.987325, len=1026.400000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[287.591273, 293.393597]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 503.950706 | 99.2% | 99.2% | 100.0% |
| angle_penalty | -3.287244 | -0.6% | 0.6% | 100.0% |
| vert_penalty | -0.831285 | -0.2% | 0.2% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
控制一个两足机器人在不平坦地形上向前行走，尽量走得远（距离）、走得快（速度），同时最小化能耗。机器人需要通过协调双腿的髋关节和膝关节力矩，产生稳定的双足步态。一旦身体倾倒，回合即结束。

## 3. 观察空间 observation_space
- type: Box (连续)
- shape: (24,)
- dtype: float32（通常）
- obs[0]: hull_angle – 躯干相对竖直方向的倾角
- obs[1]: hull_angular_velocity – 躯干角速度
- obs[2]: horizontal_velocity – 质心水平（前向）速度
- obs[3]: vertical_velocity – 质心垂直速度
- obs[4]: hip1_angle – 第一腿髋关节角度
- obs[5]: hip1_speed – 第一腿髋关节角速度
- obs[6]: knee1_angle – 第一腿膝关节角度
- obs[7]: knee1_speed – 第一腿膝关节角速度
- obs[8]: leg1_contact – 第一腿是否触地（1.0 触地，0.0 不触地）
- obs[9]: hip2_angle – 第二腿髋关节角度
- obs[10]: hip2_speed – 第二腿髋关节角速度
- obs[11]: knee2_angle – 第二腿膝关节角度
- obs[12]: knee2_speed – 第二腿膝关节角速度
- obs[13]: leg2_contact – 第二腿是否触地（1.0 触地，0.0 不触地）
- obs[14] ~ obs[23]: lidar_0 ~ lidar_9 – 10个激光雷达测距值，测量前方地形距离

## 4. 动作空间 action_space
- type: Box (连续)
- 动作数量: 4 维
- 每维范围: [-1.0, 1.0]（连续力矩值）
- action 0: hip_torque_leg1 – 第一腿髋关节施加的力矩
- action 1: knee_torque_leg1 – 第一腿膝关节施加的力矩
- action 2: hip_torque_leg2 – 第二腿髋关节施加的力矩
- action 3: knee_torque_leg2 – 第二腿膝关节施加的力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形终点，完成行走任务）
- failure-like termination: body_fallen_over（身体倾倒，行走失败）
- ambiguous termination: 无
- truncation: 未在给定代码中体现（可能存在最大步数截断，但未明确提供）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，terminated 仅由布尔值给出，无显式成功/失败标志字段）
- explicit_failure_flag_available: false
- allowed_info_fields: 无（info 为空字典）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；无法确定成功/失败原因细分

## 7. 可用于奖励函数的信号
- 躯干倾角：obs[0] (hull_angle) – 可用于衡量平衡，偏离直立越大可能越差
- 躯干角速度：obs[1] (hull_angular_velocity) – 联合倾角判断稳定性
- 水平速度：obs[2] (horizontal_velocity) – 前进速度，鼓励快速向前
- 垂直速度：obs[3] (vertical_velocity) – 可用于惩罚剧烈上下跳动
- 关节角度/角速度：obs[4]~obs[7] 和 obs[9]~obs[12] – 可用于能量惩罚（例如动作平方和、关节力矩做功近似）
- 触地标志：obs[8], obs[13] – 可用于判断步态交替
- 动作本身：action (4 维) – 可直接计算控制能量（如 sum(action^2)）
- 下一时刻的水平速度：next_obs[2] – 可用于衡量速度变化
- LIDAR 测距：obs[14]~obs[23] – 可用于鼓励避开近距离障碍、保持安全距离
- 终止相关推断：既无显式成功/失败标志，也不应直接依赖终止状态（因 reward 在终止前步计算）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_penalty + forward_reward + vert_penalty | 289.99 | 289.99 | 0.00 | 1026.40 | angle_penalty=-0.015 forward_reward=0.333 vert_penalty=-0.001 | new_best |
