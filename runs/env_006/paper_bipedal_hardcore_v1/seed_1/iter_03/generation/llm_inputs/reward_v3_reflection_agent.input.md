# 1. Search objective
- target_score: 300.000000
- current_score: -50.598476
- gap_to_target: 350.598476
- target_achievement_ratio: -16.866%

# 2. 上一轮奖励函数代码（该轮得分: -50.598476）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主进展：水平速度正向奖励（保持原样）
    horizontal_speed = obs[2]
    progress_reward = 1.0 * horizontal_speed

    # 稳定约束：躯干倾斜角 hinge 惩罚（保持原样）
    hull_angle = obs[0]
    angle_threshold = 0.3
    hinge = max(0.0, abs(hull_angle) - angle_threshold)
    stability_penalty = -0.5 * hinge

    # 新增：躯干角速度二次惩罚（早期摔倒预警）
    hull_angular_velocity = obs[1]
    angular_velocity_penalty = -0.1 * (hull_angular_velocity ** 2)

    # 效率代价：动作二次惩罚（保持原样）
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    total_reward = progress_reward + stability_penalty + angular_velocity_penalty + action_efficiency_penalty
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_efficiency_penalty': action_efficiency_penalty
    }
    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 411.40 | -18.01 | ✅ |
| 2 | 惩罚快速角速度将使策略学习避免剧烈旋转，减少摔倒概率，从而提升真实环境得分。 | 惩罚快速角速度将使策略学习避免剧烈旋转，减少摔倒概率，从而提升真实环境得分。 | 222.20 | -50.60 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-50.598476, len=222.200000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-92.283326, 39.596697]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_reward | 87.368240 | 93.7% | 94.4% | 100.0% |
| stability_penalty | -2.701581 | -2.9% | 2.9% | 15.8% |
| action_efficiency_penalty | -2.470366 | -2.6% | 2.6% | 100.0% |
| angular_velocity_penalty | -0.025876 | -0.0% | 0.0% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
双足机器人需要在布满障碍（阶梯、树桩、坑洼等）的粗糙地形上持续向前行走，尽可能走得远且高效。  
主要目标是**稳定前进**并避免摔倒；次要目标包括**最小化关节力矩消耗**和**最终抵达地形末端**。  
机器人可利用前方的激光雷达（LiDAR）感知地形，提前调整步态。  
到达地形末端会正常结束，摔倒则提前失败。

## 3. 观察空间 observation_space
- type: Box  
- shape: [24]  
- dtype: 根据 float32 推断  
- 各维度含义及 reward 可用性：

| 索引 | 名称                      | 含义                           | reward_usable |
|------|---------------------------|--------------------------------|---------------|
| 0    | hull_angle                | 躯干倾斜角                     | true          |
| 1    | hull_angular_velocity     | 躯干角速度                     | true          |
| 2    | horizontal_speed           | 质心水平速度                   | true          |
| 3    | vertical_speed             | 质心垂直速度                   | true          |
| 4    | joint_0_angle （hip_1）    | 髋关节1角度                    | true          |
| 5    | joint_0_speed             | 髋关节1角速度                  | true          |
| 6    | joint_1_angle （knee_1）   | 膝关节1角度                    | true          |
| 7    | joint_1_speed             | 膝关节1角速度                  | true          |
| 8    | joint_2_angle （hip_2）    | 髋关节2角度                    | true          |
| 9    | joint_2_speed             | 髋关节2角速度                  | true          |
| 10   | joint_3_angle （knee_2）   | 膝关节2角度                    | true          |
| 11   | joint_3_speed             | 膝关节2角速度                  | true          |
| 12   | leg_1_ground_contact      | 左腿触地指示（二值）           | true          |
| 13   | leg_2_ground_contact      | 右腿触地指示（二值）           | true          |
| 14–23| lidar_1…lidar_10          | 前方10个LiDAR测距（地形高度）   | 谨慎使用      |

- 注意：LiDAR原始数值是距离测量值，可用于隐式学习地形应对，但不建议直接作为稠密奖励信号，因为其语义与前进或平衡无直接线性关系。

## 4. 动作空间 action_space
- type: Box  
- shape: [4]  
- 范围: [-1.0, 1.0]  
- 每个动作维度含义：
  - action[0]: hip_1_torque – 第一个髋关节力矩  
  - action[1]: knee_1_torque – 第一个膝关节力矩  
  - action[2]: hip_2_torque – 第二个髋关节力矩  
  - action[3]: knee_2_torque – 第二个膝关节力矩  
- 连续力矩控制，无离散动作。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain – 抵达地形末端，环境正常结束。
- failure-like termination: body_fallen_over – 身体倾倒（典型失败）。
- ambiguous termination: 无。
- truncation: 无明显时间截断（原文未提及 max steps，但可能存在于环境中，视为 ambiguous）。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: **false** （info 字段为空，不允许使用）
- explicit_failure_flag_available: **false**
- allowed_info_fields: []  （info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（因为不允许使用任何 info 内容）
- 终止原因只能通过以下方式**推断**（derived_possible）：
  - **摔倒推断**：终止时 `next_obs[0]`（hull_angle）很可能超过阈值（如>0.8 rad），或者两腿触地指示同时为0且躯干姿态异常。可利用 `next_obs` 在 reward 中检测。
  - **到达终点推断**：终止时 `next_obs` 的 hull_angle 较小且无异常，但无法从观测直接区分；因为无位置信息，可通过 episode 忽然结束且未触发摔倒检测来判断。奖励函数设计中可仅通过前进速度奖励覆盖此目标，避免依赖显式到达奖励。

# 7. Formula switching guide
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2`，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |
| 缺少灾难性失败信号 | 终止率高且失败回合 reward 非负 | terminal_event | 从观测推断失败状态，加入硬覆盖惩罚 |
| 缺少任务完成信号 | agent 持续前进但 episode 在无摔倒情况下终止 | terminal_event 或 improvement_delta | 用位置 delta 做正向奖励，或在确认可达终点时加入软完成 bonus |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 8. 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | action_efficiency_penalty + progress_reward + stability_penalty | -18.01 | -18.01 | 0.00 | 411.40 | action_efficiency_penalty=-0.018 progress_reward=0.218 stability_penalty=-0.010 | new_best |
| 2 | action_efficiency_penalty + angular_velocity_penalty + progress_reward + stability_penalty | -50.60 | -18.01 | -32.59 | 222.20 | action_efficiency_penalty=-0.018 angular_velocity_penalty=-0.000 progress_reward=0.202 stability_penalty=-0.008 | no_meaningful_improvement |
