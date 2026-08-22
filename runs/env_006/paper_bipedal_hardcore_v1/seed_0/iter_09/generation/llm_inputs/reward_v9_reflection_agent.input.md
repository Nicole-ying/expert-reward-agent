# 1. Search objective
- target_score: 300.000000
- current_score: -50.616590
- gap_to_target: 350.616590
- target_achievement_ratio: -16.872%

# 2. 上一轮奖励函数代码（该轮得分: -50.616590）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    # obs 索引 (共24维)
    hull_angle = obs[0]           # 躯干倾角 (rad)
    hull_ang_vel = obs[1]         # 躯干角速度 (rad/s)
    horizontal_speed = obs[2]     # 水平速度 (m/s)
    vertical_speed = obs[3]       # 垂直速度 (m/s)
    # obs[4..11]: 关节角度和角速度，暂不直接使用
    leg_1_contact = obs[12]       # 腿1触地 (0/1)
    leg_2_contact = obs[13]       # 腿2触地 (0/1)
    # obs[14..23]: lidar, 不可用于奖励

    # next_obs 索引
    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_vertical_speed = next_obs[3]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    #    role: forward_velocity_reward
    #    formula: dense_state_signal (线性), 直接鼓励保持正水平速度
    #    scale: 1.0 使每步贡献约 0.8~2.0
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)  # 负向速度不奖励
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门 (soft_health_gate)
    #    role: upright_penalty 变形为 gate, 替代独立惩罚
    #    rationale: 尝试过的独立 tilt penalty 效果不佳 (score -49~-62).
    #               采用 soft gate 在姿态恶化时直接衰减 forward_reward,
    #               避免 agent 在倾斜时仍因高速获得大量奖励。
    #    formula: soft_health_gate (线性衰减)
    #    gate = 1.0 当 hull_angle 在安全范围内,
    #           线性衰减至 0.0 当 hull_angle 接近危险阈值
    # =========================================================================
    # 设定安全区和衰退区间
    tilt_safe_bound = 0.3          # rad, 近似17°, 正常行走摆动范围
    tilt_danger_bound = 0.7        # rad, 近似40°, 接近摔倒临界 (经验阈值 ~0.8)
    tilt_margin = tilt_danger_bound - tilt_safe_bound  # 0.4 rad 衰退区间

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    # 角速度惩罚: 当躯干快速旋转时进一步收紧 gate, 捕捉突然失去平衡的前兆
    # 在 tilt_gate 基础上再乘一个角速度衰减因子
    ang_vel_thresh = 2.0           # rad/s, 正常步态摆动通常 < 1.5
    ang_vel_margin = 4.0           # 2.0~6.0 rad/s 区间衰减
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3       # 不归零, 保留微弱梯度以防止完全丧失学习信号
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 能量效率惩罚 (轻量)
    #    role: energy_penalty
    #    formula: action_efficiency (L2 范数)
    #    scale: 极小权重, 仅在主任务已驱动后提供效率偏好
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum  # 步贡献约 -0.005~-0.02

    # =========================================================================
    # 5. 组合并返回
    # =========================================================================
    # 核心思路: 主奖励通过稳定性 gate 衰减, 再减去轻量能耗
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward - energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "energy_penalty": -energy_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 166.90 | -57.47 | ✅ |
| 2 | 改用二次惩罚后，即使小角度也会产生可感知的 penalty，训练会学会将 tilt 和 angvel 维持在低水平... | 改用二次惩罚后，即使小角度也会产生可感知的 penalty，训练会学会将 tilt 和 angvel 维持在低水平... | 167.40 | -62.54 | ❌ |
| 3 | Hinge 惩罚在倾角超出安全阈值后立即给出线性增长的代价，使 agent 能更早学会“维持小倾角”，减少摔倒次数... | Hinge 惩罚在倾角超出安全阈值后立即给出线性增长的代价，使 agent 能更早学会“维持小倾角”，减少摔倒次数... | 163.65 | -49.54 | ✅ |
| 4 | 加入角速度 gate 后，即将摔倒时的 forward reward 会被提前关断，reward hack 失效，... | 加入角速度 gate 后，即将摔倒时的 forward reward 会被提前关断，reward hack 失效，... | 163.65 | -49.54 | ❓ |
| 5 | 骨架变化: action_cost + contact_transition_reward + forward_ | — | 406.50 | -42.42 | ✅ |
| 6 | 将 contact_reward 乘以 forward_speed 后，agent 只有在真正前进时才能获得步态激... | 将 contact_reward 乘以 forward_speed 后，agent 只有在真正前进时才能获得步态激... | 359.70 | -62.76 | ❌ |
| 7 | 恢复 contact_reward 的独立性后，步态切换激励回归，配合 forward_reward 引导前进，分... | 恢复 contact_reward 的独立性后，步态切换激励回归，配合 forward_reward 引导前进，分... | 757.95 | -43.01 | ❓ |
| 8 | 骨架变化: energy_penalty + forward_reward + gated_forward +  | — | 230.80 | -50.62 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-50.616590, len=230.800000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-93.489156, -8.236010]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stability_gate | 222.021689 | 55.7% | 55.7% | 99.7% |
| forward_reward | 89.222591 | 22.4% | 22.4% | 97.7% |
| gated_forward | 85.906493 | 21.5% | 21.5% | 97.4% |
| energy_penalty | -1.554871 | -0.4% | 0.4% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境中的双足机器人需要在不规则的粗糙地形上尽可能向前行进，同时保持能量效率。地形包含阶梯状、树桩、坑洞等变化，因此机体必须利用前方 10 个 LIDAR 测距信号来预判地形，动态调整步态。主要目标是：稳定行走、远离摔倒、尽量走远；次要目标是：最小化不必要的关节扭矩（能耗）。该任务的核心是崎岖地形上的持续运动控制，而非单纯到达指定坐标点。

## 3. 观察空间 observation_space
- type: Box  
- shape: (24,)  
- dtype: float32（默认推断）  
- obs 各维含义：

| index | 名称                     | 含义                                   | reward_usable |
|-------|--------------------------|----------------------------------------|---------------|
| 0     | hull_angle               | 身体基座倾角                           | true          |
| 1     | hull_angular_velocity    | 身体基座角速度                         | true          |
| 2     | horizontal_speed         | 质心水平速度                           | true          |
| 3     | vertical_speed           | 质心垂直速度                           | true          |
| 4     | joint_0_angle (hip_1)    | 髋关节 1 角度                          | true          |
| 5     | joint_0_speed (hip_1)    | 髋关节 1 角速度                        | true          |
| 6     | joint_1_angle (knee_1)   | 膝关节 1 角度                          | true          |
| 7     | joint_1_speed (knee_1)   | 膝关节 1 角速度                        | true          |
| 8     | joint_2_angle (hip_2)    | 髋关节 2 角度                          | true          |
| 9     | joint_2_speed (hip_2)    | 髋关节 2 角速度                        | true          |
| 10    | joint_3_angle (knee_2)   | 膝关节 2 角度                          | true          |
| 11    | joint_3_speed (knee_2)   | 膝关节 2 角速度                        | true          |
| 12    | leg_1_ground_contact     | 腿 1 是否接地（0 或 1）                | true          |
| 13    | leg_2_ground_contact     | 腿 2 是否接地（0 或 1）                | true          |
| 14    | lidar_1                  | 第一根 LIDAR 测距值（前方地形高度）    | true          |
| 15    | lidar_2                  | 第二根 LIDAR 测距值                    | true          |
| 16    | lidar_3                  | 第三根 LIDAR 测距值                    | true          |
| 17    | lidar_4                  | 第四根 LIDAR 测距值                    | true          |
| 18    | lidar_5                  | 第五根 LIDAR 测距值                    | true          |
| 19    | lidar_6                  | 第六根 LIDAR 测距值                    | true          |
| 20    | lidar_7                  | 第七根 LIDAR 测距值                    | true          |
| 21    | lidar_8                  | 第八根 LIDAR 测距值                    | true          |
| 22    | lidar_9                  | 第九根 LIDAR 测距值                    | true          |
| 23    | lidar_10                 | 第十根 LIDAR 测距值                    | true          |

注：接地信号为 0/1 标量，间接反映了支撑相，可用于步态激励或摔倒检测。

## 4. 动作空间 action_space
- type: Box  
- shape: (4,)  
- bounds: [-1.0, 1.0]  
- 各维含义：
  - action_dim 0: hip_1_torque，第一髋关节力矩
  - action_dim 1: knee_1_torque，第一膝关节力矩
  - action_dim 2: hip_2_torque，第二髋关节力矩
  - action_dim 3: knee_2_torque，第二膝关节力矩

所有动作均为连续值，力矩限幅在 [-1, 1] 内。

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: reached_end_of_terrain（到达地形末端，视为成功）
- failure-like termination: body_fallen_over（机体摔倒）
- ambiguous termination: 无
- truncation: 无时间上限截断（隐含 episode 可能在短步数内因摔倒而终止，但未明确提供 truncation 信号）

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false （info 为空，无法直接读取 success 标志）
- explicit_failure_flag_available: false （同上，无直接 failure 字段）
- allowed_info_fields: 无（info 字典为空）
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用；original_reward 被强制屏蔽

终止条件的判断只能通过观测间接进行：
- 摔倒：可依据 hull_angle 绝对值超过某经验阈值（如 >0.5 rad）且可能伴随 vertical_speed 突变或 leg 接触异常；标记为 derived_possible。
- 到达终点：在 episode 结束时若 terminated=True 且未检测到摔倒，可推测为成功。但 compute_reward 中无法直接获取 terminated 标志，只能通过最后一步的 next_obs 状态推测，存在误判风险。

因此，成功/失败的信号是弱可用的，理想情况下应避免依赖终点信号，而是专注于持续前进和生存的激励。

## 7. 可用于奖励函数的信号
- position: 无直接位置（但 horizontal_speed 可积分得到水平位移增量）；垂直位移可从 vertical_speed 累积或间接通过高度变化推断（但无绝对高度观测）。
- velocity: horizontal_speed (obs[2])，vertical_speed (obs[3])，各关节角速度 (obs[5,7,9,11])
- orientation: hull_angle (obs[0])，hull_angular_velocity (obs[1])
- contact: leg_1_ground_contact (obs[12])，leg_2_ground_contact (obs[13])，二值信号，用于检测支撑相或摔倒（例如连续若干步双脚未接地即可能摔倒）。
- action/engine: action 四维力矩（hip_1, knee_1, hip_2, knee_2），可直接用于扭矩惩罚。
- other: LIDAR 读数 (obs[14:24])，提供地形预览，可用于鼓励预判性步态调整，但不易直接转化为标量奖赏，通常用于辅助特征而非独立 reward 项；也可用于检测极端地形。

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
| 1 | gated_forward_speed + posture_hinge_penalty | -57.47 | -57.47 | 0.00 | 166.90 | gated_forward_speed=0.238 posture_hinge_penalty=-0.009 | new_best |
| 2 | gated_forward_speed + stability_quad_penalty | -62.54 | -57.47 | -5.07 | 167.40 | gated_forward_speed=0.164 stability_quad_penalty=-0.042 | no_meaningful_improvement |
| 3 | gated_forward_speed + stability_tilt_hinge_penalty | -49.54 | -49.54 | 0.00 | 163.65 | gated_forward_speed=0.232 stability_tilt_hinge_penalty=-0.007 | new_best |
| 4 | gated_forward_speed + stability_tilt_hinge_penalty | -49.54 | -49.54 | 0.00 | 163.65 | gated_forward_speed=0.232 stability_tilt_hinge_penalty=-0.007 | unsolved_stagnation_fresh_restart |
| 5 | action_cost + contact_transition_reward + forward_reward_gated | -42.42 | -42.42 | 0.00 | 406.50 | action_cost=-0.018 contact_transition_reward=0.097 forward_reward_gated=0.239 | new_best |
| 6 | action_cost + contact_transition_reward + forward_reward_gated | -62.76 | -42.42 | -20.34 | 359.70 | action_cost=-0.019 contact_transition_reward=0.049 forward_reward_gated=0.256 | no_meaningful_improvement |
| 7 | action_cost + contact_transition_reward + forward_reward_gated | -43.01 | -42.42 | -0.59 | 757.95 | action_cost=-0.018 contact_transition_reward=0.246 forward_reward_gated=0.155 | unsolved_stagnation_fresh_restart |
| 8 | energy_penalty + forward_reward + gated_forward + stability_gate | -50.62 | -42.42 | -8.20 | 230.80 | energy_penalty=-0.010 forward_reward=0.247 gated_forward=0.233 stability_gate=0.927 | no_meaningful_improvement |
