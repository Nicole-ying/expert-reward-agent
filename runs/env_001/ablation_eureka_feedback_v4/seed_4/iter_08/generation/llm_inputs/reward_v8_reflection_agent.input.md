# Search objective
- target_score: 200.000000
- current_score: 256.347342
- gap_to_target: -56.347342
- target_achievement_ratio: 128.174%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 256.347342）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Current and next positions relative to landing pad center
    dx_curr, dy_curr = obs[0], obs[1]
    dx_next, dy_next = next_obs[0], next_obs[1]
    dist_curr = (dx_curr**2 + dy_curr**2) ** 0.5
    dist_next = (dx_next**2 + dy_next**2) ** 0.5

    # Approach progress: positive when getting closer to the pad
    approach_delta = dist_curr - dist_next

    # Next‑step velocity and safety threshold
    vx_next, vy_next = next_obs[2], next_obs[3]
    speed_next = (vx_next**2 + vy_next**2) ** 0.5
    safe_speed = 0.2

    # Next‑step orientation and angular velocity
    angle_next = next_obs[4]
    angvel_next = next_obs[5]

    # Contact flags
    left_contact_next = next_obs[6] > 0.5
    right_contact_next = next_obs[7] > 0.5

    # --- Reward weights (best skeleton) ---
    w_approach = 2.0
    w_vel_penalty = 0.5
    w_angle = 0.5
    w_angvel = 0.1
    w_landing = 0.1
    w_grounded = 0.3        # new continuous grounded quality weight

    # Component 1: dense progress towards the pad
    approach_reward = w_approach * approach_delta

    # Component 2: speed constraint (hinge)
    vel_penalty = -w_vel_penalty * max(0.0, speed_next - safe_speed)

    # Component 3: angular stability (quadratic penalties)
    angle_stability = -w_angle * (angle_next**2) - w_angvel * (angvel_next**2)

    # Component 4: continuous landing‑quality proxy (unchanged)
    dist_factor = 1.0 / (1.0 + 1.0 * dist_next)
    speed_factor = 1.0 / (1.0 + 1.0 * speed_next)
    landing_reward = w_landing * dist_factor * speed_factor

    # Component 5 (REPLACED): grounded quality — only when leg contact and low speed
    grounded_reward = 0.0
    if left_contact_next or right_contact_next:
        # linear factor: 1 at zero speed, 0 at safe_speed, 0 above
        speed_ratio = max(0.0, 1.0 - speed_next / safe_speed)
        grounded_reward = w_grounded * speed_ratio

    total_reward = approach_reward + vel_penalty + angle_stability + landing_reward + grounded_reward

    components = {
        "approach_progress": approach_reward,
        "velocity_penalty": vel_penalty,
        "angle_stability": angle_stability,
        "landing_quality": landing_reward,
        "grounded_quality": grounded_reward,
    }
    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=256.347342, len=351.350000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[229.129917, 291.378163]

## Reward component values (mean per episode)
- landing_quality: 20.770152
- grounded_quality: 13.972010
- velocity_penalty: -10.721436
- approach_progress: 2.745139
- angle_stability: -2.141589

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 **2D 类飞行器轨迹优化任务**。  
主体从一个靠近顶部中央的随机位置出发，带有初始随机作用力。  
核心目标是 **到达中央的目标着陆垫并保持稳定停靠（settle）**，同时尽可能 **快速** 且 **消耗最少的引擎推力**。  
智能体需要学会接近目标、减小速度、保持竖直姿态并安全接触。  
混淆目标：不要将“纯粹节约燃料”或“纯粹高速飞行”当作唯一目标，必须在安全着陆的前提下平衡时间与能耗。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断，未显式说明但符合常规）
- obs[0]: x_position — 相对目标垫中心的水平坐标，可用于奖励函数 (reward_usable: true)
- obs[1]: y_position — 相对目标垫高度的垂直坐标，可用于奖励函数 (reward_usable: true)
- obs[2]: x_velocity — 水平速度，可用于奖励函数 (reward_usable: true)
- obs[3]: y_velocity — 垂直速度，可用于奖励函数 (reward_usable: true)
- obs[4]: body_angle — 机体朝向角，可用于奖励函数 (reward_usable: true)
- obs[5]: angular_velocity — 角速度，可用于奖励函数 (reward_usable: true)
- obs[6]: left_support_contact — 左侧支撑腿接触标志（0或1），可用于奖励函数 (reward_usable: true)
- obs[7]: right_support_contact — 右侧支撑腿接触标志（0或1），可用于奖励函数 (reward_usable: true)

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action 0: no_engine — 不施加推力
- action 1: left_orientation_engine — 启动左方向引擎，通常产生旋转力矩（可能伴随微小侧向力）
- action 2: main_engine — 启动主引擎，产生向上的推力
- action 3: right_orientation_engine — 启动右方向引擎，通常产生反向旋转力矩

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination:  
  `body_not_awake_or_settled` — 机体稳定不活动且可能已停靠在垫子上。该条件本身不一定表明成功，但若同时位置靠近目标垫，则极大概率是成功着陆。
- failure-like termination:  
  `crash_or_body_contact` — 机体发生碰撞或非预期身体接触（如侧面触地），可能是坠毁。  
  `horizontal_position_outside_viewport` — 水平坐标超出视口边界，视为飞出有效区域，失败。
- ambiguous termination:  
  上述条件被合并为一个布尔标志，无法在 `info` 中直接区分终止原因。
- truncation: 无显式截断（step 返回 `False` for truncation），只有终止。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 无（step 返回空字典 `{}`），因此禁止使用任何 info 字段。
- forbidden_or_uncertain_info_fields: 所有 info 字段（`success`, `failure`, `termination_reason` 等）均不可用。

## 7. 可用于奖励函数的信号
- **position**：`obs[0]`(x相对位置)，`obs[1]`(y相对位置) —— 可构建到目标垫的距离、高度偏差。
- **velocity**：`obs[2]`(x速度)，`obs[3]`(y速度) —— 可构建总速度或速度分量，用于鼓励低速着陆。
- **orientation**：`obs[4]`(角度)，`obs[5]`(角速度) —— 可促使保持竖直。
- **contact**：`obs[6]`(左接触)，`obs[7]`(右接触) —— 可评估是否已着陆或着陆质量。
- **action/engine**：动作本身以及 `action != 0` 的布尔值 —— 可直接用于惩罚引擎使用。
- **other**：可利用 `next_obs` 与 `obs` 的差分（如位置变化、接触变化）来检测状态转移，但不推荐依赖复杂构造。

# Compact expert route context
# Formula switching guide (evidence → operator)
| 当前形态 | 证据模式 | 目标算子 | 变换要点 |
|---|---|---|---|
| 线性正奖励 `w * signal` | score 停滞在低水平，signal 正值但偏小 | dense_state_signal (凸化) | 改用 `signal**2` 或指数形式，保持系数使量级可比 |
| 全时二次惩罚 `-w * error**2` | 惩罚 active_rate≈100% 但 terminated 率仍高 | dense_state_signal (hinge) | 改 `max(0, threshold - signal)`，threshold 设在终止边界的 60-80% |
| 独立约束惩罚 + 高 terminated | terminated 主因是某状态越界，惩罚已加但无效 | soft_health_gate | 把该状态做成 gate 乘到主奖励上，不额外增加独立惩罚 |
| 稀疏二值 proxy | active_rate < 5%，episode 很短 | joint_condition_proxy (连续化) | 把二值条件换成连续 bounded factor，确保每步有梯度 |
| 乘积 proxy 经常塌缩为 0 | 多个 factor 中总有一个趋近 0 | joint_condition_proxy (几何平均) | 用 `(f1 * f2 * ...) ** (1/n)` 替代裸乘积 |

Key anti-patterns: prefer gate over bigger penalty; prefer hinge over quadratic for boundary constraints; convexify forward reward when stuck at low-speed plateau.

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | angle_stability + approach_progress + landing_quality + velocity_penalty | -19.32 | -19.32 | 0.00 | 1000.00 | angle_stability=-0.041 approach_progress=0.007 landing_quality=1.523 velocity_penalty=-0.946 | new_best |
| 2 | angle_stability + approach_progress + landing_quality + velocity_penalty | -37.99 | -19.32 | -18.66 | 993.05 | angle_stability=-0.048 approach_progress=0.008 landing_quality=1.610 velocity_penalty=-1.083 | no_meaningful_improvement |
| 3 | angle_stability + approach_progress + landing_quality + velocity_penalty | 177.41 | 177.41 | 0.00 | 652.25 | angle_stability=-0.032 approach_progress=0.006 landing_quality=0.062 velocity_penalty=-0.734 | new_best |
| 4 | angle_stability + approach_progress + landing_quality + velocity_penalty | 179.80 | 179.80 | 0.00 | 795.65 | angle_stability=-0.009 approach_progress=0.005 landing_quality=0.074 velocity_penalty=-0.048 | new_best |
| 5 | angle_stability + approach_progress + landing_quality + velocity_penalty | 89.16 | 179.80 | -90.64 | 1000.00 | angle_stability=-0.010 approach_progress=0.006 landing_quality=0.007 velocity_penalty=-0.060 | no_meaningful_improvement |
| 6 | angle_stability + approach_progress + landing_quality + touchdown_success + velocity_penalty | 24.61 | 179.80 | -155.19 | 884.35 | angle_stability=-0.021 approach_progress=0.007 landing_quality=0.062 touchdown_success=0.616 velocity_penalty=-0.094 | unsolved_high_achievement_continue_from_best |
| 7 | angle_stability + approach_progress + grounded_quality + landing_quality + velocity_penalty | 256.35 | 256.35 | 0.00 | 351.35 | angle_stability=-0.014 approach_progress=0.009 grounded_quality=0.155 landing_quality=0.068 velocity_penalty=-0.106 | target_solved_new_best |
