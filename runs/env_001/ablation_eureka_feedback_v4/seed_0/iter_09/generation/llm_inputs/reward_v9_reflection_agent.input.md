# Duplicate reward retry
The previous generation duplicated iter 8 (runs\env_001\ablation_eureka_feedback_v4\seed_0\iter_08\generation\reward_v8.py). Retry 1: generate a materially different reward function.
The previous draft is semantically identical to the previous trained reward and is not a valid search intervention. Re-analyze the full environment facts, training feedback, Agent Memory, previous reward, and best reward below. Choose a different evidence-based modification plan, then implement one concrete tune/delete/add/mix change. Return a complete reward function whose executable code is materially different from every historical reward. Do not merely rename variables or comments.

# Rejected duplicate draft
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # current position
    x0 = obs[0]
    y0 = obs[1]
    # next position
    x1 = next_obs[0]
    y1 = next_obs[1]

    dist0 = (x0**2 + y0**2) ** 0.5
    dist1 = (x1**2 + y1**2) ** 0.5
    delta = dist0 - dist1  # >0 when approaching pad
    progress = 5.0 * max(0.0, delta)

    # next state
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward: base contact + bonus for both legs contacting while stable
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

    joint = (stability * progress + 1e-12) ** 0.5

    # small engine penalty to discourage unnecessary thrust
    engine_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total = joint + contact + engine_penalty

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact,
        "engine_penalty": engine_penalty,
    }
    return float(total), components
```

# Search objective
- target_score: 200.000000
- current_score: 239.517348
- gap_to_target: -39.517348
- target_achievement_ratio: 119.759%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 239.517348）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # current position
    x0 = obs[0]
    y0 = obs[1]
    # next position
    x1 = next_obs[0]
    y1 = next_obs[1]

    dist0 = (x0**2 + y0**2) ** 0.5
    dist1 = (x1**2 + y1**2) ** 0.5
    delta = dist0 - dist1  # >0 when approaching pad
    progress = 5.0 * max(0.0, delta)

    # next state
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    speed_sq = vx**2 + vy**2
    stability = 1.0 / (1.0 + 0.1 * speed_sq + 0.04 * (ang_vel**2) + 1.0 * (angle**2))

    # contact reward: base contact + bonus for both legs contacting while stable
    contact = 0.5 * (left_contact + right_contact) + 0.2 * left_contact * right_contact * stability

    joint = (stability * progress + 1e-12) ** 0.5

    # small engine penalty to discourage unnecessary thrust
    engine_penalty = -0.03 * (1.0 if action != 0 else 0.0)

    total = joint + contact + engine_penalty

    components = {
        "progress": progress,
        "stability": stability,
        "contact": contact,
        "engine_penalty": engine_penalty,
    }
    return float(total), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback (EUREKA-style)

## Final-policy outcome

score=239.517348, len=387.900000, terminated=19/20, truncated=1/20, reward_errors=0
score_range=[113.995736, 270.018272]

## Reward component values (mean per episode)
- stability: 382.118864
- contact: 96.029143
- engine_penalty: -8.973000
- progress: 6.637641

# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 着陆器轨迹优化任务。主体（着陆器）从视角顶部中央附近出发，受初始随机作用力，目标是以最快速度到达画面中心的着陆垫并稳定停靠（settle），同时尽可能少地使用引擎推力。智能体需要学会：  
- 精确接近目标垫（将相对位置向量驱近于零）  
- 减速至软接触（降低线速度与角速度，保持姿态水平）  
- 利用两条支撑腿与垫形成安全、稳定的接触  
- 附属优化：降低燃料消耗（减少引擎动作）和减少用时  

该任务不应被混淆为纯粹的平衡、抓取或多足行走；其核心是到达并稳定停留在指定目标位置。

## 3. 观察空间 observation_space
- type: Box  
- shape: (8,)  
- dtype: float64  
- obs[0]: x_position – 相对于着陆垫的水平坐标，reward_usable: true  
- obs[1]: y_position – 相对于垫高度的垂直坐标，reward_usable: true  
- obs[2]: x_velocity – 水平线速度，reward_usable: true  
- obs[3]: y_velocity – 垂直线速度，reward_usable: true  
- obs[4]: body_angle – 主体姿态角（方向），reward_usable: true  
- obs[5]: angular_velocity – 角速度，reward_usable: true  
- obs[6]: left_support_contact – 左支撑腿是否触垫（0.0或1.0），reward_usable: true  
- obs[7]: right_support_contact – 右支撑腿是否触垫（0.0或1.0），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete  
- n: 4  
- action 0: no_engine – 不点火，依靠当前速度和重力演化  
- action 1: left_orientation_engine – 点燃左侧姿态引擎（产生扭矩，逆时针？）  
- action 2: main_engine – 点燃主发动机（产生向上的推力，可能同时影响姿态）  
- action 3: right_orientation_engine – 点燃右侧姿态引擎（产生反向扭矩，顺时针？）

## 5. step 与终止条件分析
### 5.1 终止模式
- success-like termination: body_not_awake_or_settled（身体不再活跃/已稳定停靠）——若同时观察到 left_contact 且 right_contact 且速度/角速度极低，很可能为成功着陆。但 step 源码中 info 为空，不能直接作为显式成功标志。  
- failure-like termination: crash_or_body_contact（除垫外与其他表面碰撞，或非预期身体接触），以及 horizontal_position_outside_viewport（水平超出视口）。这两种情况可视为着陆失败。  
- ambiguous termination: body_not_awake_or_settled 可能因物理休眠触发，但需要结合接触和速度才能确认成功；若未触垫却休眠（例如悬停在半空停止计算）应视为异常终止，不可直接用作奖励依据。  
- truncation: 无时间截断迹象（源码未显示步骤限制），但实际环境中可能存在。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false（info 为空，没有 success 字段）  
- explicit_failure_flag_available: false  
- allowed_info_fields: （空）  
- forbidden_or_uncertain_info_fields: info 中任何未声明的字段均不可使用；严禁假设 info["success"]、info["failure"]、info["termination_reason"] 等存在。

## 7. 可用于奖励函数的信号
- position: 相对目标垫的水平和垂直位置 (next_obs[0], next_obs[1])  
- velocity: 水平和垂直线速度 (next_obs[2], next_obs[3])  
- orientation: 姿态角 (next_obs[4])  
- angular velocity: 角速度 (next_obs[5])  
- contact: 左右支撑触垫标志 (next_obs[6], next_obs[7])  
- action/engine: 动作选择（0-3），可用于燃料消耗或动作惩罚  
- other: 无

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
| 1 | contact + proximity + stability | -109.81 | -109.81 | 0.00 | 68.40 | contact=0.024 proximity=-0.972 stability=-0.080 | new_best |
| 2 | contact + proximity + stability | -73.53 | -73.53 | 0.00 | 69.75 | contact=0.025 proximity=-0.966 stability=-0.159 | new_best |
| 3 | contact + proximity + stability | -582.65 | -73.53 | -509.11 | 65.25 | contact=0.017 proximity=-1.001 stability=0.442 | no_meaningful_improvement |
| 4 | contact + progress + stability | 46.22 | 46.22 | 0.00 | 803.90 | contact=0.493 progress=0.721 stability=0.961 | new_best |
| 5 | contact + progress + stability | -14.18 | 46.22 | -60.40 | 1000.00 | contact=0.467 progress=0.709 stability=0.958 | no_meaningful_improvement |
| 6 | contact + progress + stability | 110.89 | 110.89 | 0.00 | 1000.00 | contact=0.581 progress=0.012 stability=0.969 | new_best |
| 7 | contact + progress + stability | 145.22 | 145.22 | 0.00 | 836.05 | contact=0.672 progress=0.012 stability=0.961 | new_best |
| 8 | contact + engine_penalty + progress + stability | 239.52 | 239.52 | 0.00 | 387.90 | contact=0.637 engine_penalty=-0.022 progress=0.014 stability=0.955 | target_solved_new_best |
