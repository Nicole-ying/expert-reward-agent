# 上一轮奖励函数代码（该轮得分: 272.072089）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward function for 2D bipedal locomotion on rough terrain.
    Drives forward progress while maintaining stable posture.
    No explicit success/failure flags are available in info.
    """
    # ---------- forward progress component ----------
    # Horizontal velocity in the forward direction (next_obs[2]).
    # Positive values correspond to moving forward, negative to backward.
    forward_velocity = next_obs[2]
    w_fwd = 1.0
    forward_reward = w_fwd * forward_velocity

    # ---------- stability penalty component ----------
    # Penalize large hull tilt (deviation from upright) and vertical velocity (jumping/falling).
    # Using absolute values gives smooth, dense gradients.
    hull_angle = next_obs[0]          # tilt angle
    vertical_velocity = next_obs[3]   # vertical speed

    w_angle = 0.5
    w_vertical = 0.1
    stability_penalty = -w_angle * abs(hull_angle) - w_vertical * abs(vertical_velocity)

    # ---------- total reward ----------
    total_reward = forward_reward + stability_penalty

    # components dict: only the terms that are directly summed into total_reward
    components = {
        "forward_reward": forward_reward,
        "stability_penalty": stability_penalty
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=272.072089, len=1284.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[268.079896, 274.661439]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| forward_reward | 504.165821 | 88.7% | 88.7% | 100.0% |
| stability_penalty | -64.517472 | -11.3% | 11.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实摘要（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
这是一个 2D 双足机器人运动任务。机器人需要在崎岖地形上稳定行走，尽可能走得远、走得快，同时尽量节省能量。机器人拥有髋关节和膝关节，需要协调四条关节（双腿各一髋一膝）产生连续的步态。如果身体摔倒，回合立即终止；如果抵达地形终点，回合也终止。任务希望机器人学会快速、高效的前进步态，避免摔倒。

## 3. 观察空间 observation_space
- type: Box（连续向量）
- shape: (24,)
- dtype: 推断为 float32 或 float64
- obs[0]: hull_angle —— 主躯干相对于直立方向的角度
- obs[1]: hull_angular_velocity —— 主躯干角速度
- obs[2]: horizontal_velocity —— 水平（前进/后退）线速度
- obs[3]: vertical_velocity —— 垂直方向线速度
- obs[4]: hip1_angle —— 腿1髋关节角度
- obs[5]: hip1_speed —— 腿1髋关节角速度
- obs[6]: knee1_angle —— 腿1膝关节角度
- obs[7]: knee1_speed —— 腿1膝关节角速度
- obs[8]: leg1_contact —— 腿1触地标志（1.0 触地，0.0 离地）
- obs[9]: hip2_angle —— 腿2髋关节角度
- obs[10]: hip2_speed —— 腿2髋关节角速度
- obs[11]: knee2_angle —— 腿2膝关节角度
- obs[12]: knee2_speed —— 腿2膝关节角速度
- obs[13]: leg2_contact —— 腿2触地标志（1.0 触地，0.0 离地）
- obs[14..23]: lidar_0..lidar_9 —— 10 个激光雷达距离测量值，用于感知前方地形

## 4. 动作空间 action_space
- type: Box（连续动作）
- shape: (4,)
- 取值范围: 每维均在 [-1.0, 1.0]（关节扭矩）
- action 0: hip_torque_leg1 —— 腿1髋关节扭矩
- action 1: knee_torque_leg1 —— 腿1膝关节扭矩
- action 2: hip_torque_leg2 —— 腿2髋关节扭矩
- action 3: knee_torque_leg2 —— 腿2膝关节扭矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（抵达地形终点，通常算成功完成任务）
- failure-like termination: body_fallen_over（身体摔倒，失败）
- ambiguous termination: 无
- truncation: 无（step 返回的 terminated 标志直接表示结束，没有另行截断）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，没有 success 字段）
- explicit_failure_flag_available: false（info 为空，没有 failure 字段）
- allowed_info_fields: 无（info 为空字典 {}）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用

## 7. 可用于奖励函数的信号
- position（通过观测可间接获得位移增量）：
  - 可利用连续帧的 horizontal_velocity（obs[2] 与 next_obs[2]）计算前进距离增量
  - 但注意没有直接的位置坐标
- velocity：
  - obs[2] horizontal_velocity（前进速度）
  - obs[3] vertical_velocity（垂直速度，可用于惩罚跳跃或坠落）
- orientation：
  - obs[0] hull_angle（躯干倾角，接近 0 表示直立）
  - obs[1] hull_angular_velocity（倾摆角速度，可用于惩罚快速翻滚）
- contact：
  - obs[8] leg1_contact（触地标志）
  - obs[13] leg2_contact（触地标志）
- action/engine：
  - 动作扭矩（action[0..3]）可用于衡量能耗（例如平方和）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | forward_reward + stability_penalty | 272.07 | 272.07 | 0.00 | 1284.80 | forward_reward=0.323 stability_penalty=-0.044 | new_best |
