# Search objective
- target_score: 200.000000
- current_score: 14.056370
- gap_to_target: 185.943630
- target_achievement_ratio: 7.028%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 14.056370）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Hybrid reward: progress-driven approach + distance-gated settle + landing bonus.
    
    Key changes from failed settle_delta:
    - Restore absolute settle_reward (distance-gated, only near target)
    - Increase progress_delta weight (3→10) as primary approach driver
    - Reduce distance_penalty (0.5→0.2) to ease negative pressure
    - Reduce velocity_damping (1.5→0.6) and orientation penalty
    - Add large landing_bonus for stable terminal state (self-limiting via episode termination)
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speeds ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (for velocity/orientation damping) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # --- Distance gate for settle reward: active only near target ---
    settle_dist_gate = max(0.0, 1.0 - curr_distance / 2.0)  # 1 at target, 0 beyond dist 2.0

    # ============================================================
    # Component A: progress_delta (primary approach guidance, STRONG)
    # ============================================================
    w_progress = 10.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (gentle baseline pull)
    # ============================================================
    w_dist = 0.2
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: velocity_damping (proximity-gated, reduced)
    # ============================================================
    w_vel = 0.6
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component D: settle_reward (absolute, distance-gated)
    # ============================================================
    # Quality factors: speed, angle, contact
    speed_quality = 1.0 / (1.0 + 5.0 * curr_speed)
    angle_quality = 1.0 / (1.0 + 5.0 * abs(nangle))
    contact_score = 0.5 * (nleft_contact + nright_contact)
    settle_quality = settle_dist_gate * speed_quality * angle_quality * (1.0 + contact_score)
    w_settle = 3.0
    settle_reward = w_settle * settle_quality

    # ============================================================
    # Component E: orientation_penalty (proximity-gated, reduced)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # ============================================================
    # Component G: landing_bonus (strong terminal incentive)
    # ============================================================
    # Only when truly landed: both legs down, stable, near target
    # Episode naturally terminates soon after, preventing farming
    landing_conditions = (
        nleft_contact > 0.5 and nright_contact > 0.5 and
        curr_distance < 0.3 and curr_speed < 0.2 and abs(nangle) < 0.1
    )
    w_landing = 40.0
    landing_bonus = w_landing if landing_conditions else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + velocity_damping
        + settle_reward
        + orientation_penalty
        + engine_penalty
        + landing_bonus
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "velocity_damping": float(velocity_damping),
        "settle_reward": float(settle_reward),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
        "landing_bonus": float(landing_bonus),
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=14.056370, len=264.100000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-142.530134, 285.530217]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 298.000000 | 52.7% | 52.7% | 2.8% |
| settle_reward | 201.652282 | 35.7% | 35.7% | 100.0% |
| distance_penalty | -37.122615 | -6.6% | 6.6% | 100.0% |
| progress_delta | 7.095325 | 1.3% | 2.1% | 99.2% |
| velocity_damping | -11.224114 | -2.0% | 2.0% | 99.9% |
| engine_penalty | -4.885500 | -0.9% | 0.9% | 61.7% |
| orientation_penalty | -0.286798 | -0.1% | 0.1% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 环境事实与专家任务画像（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本任务是 **2D 飞行器着陆控制**：一个搭载主推进器与姿态推进器的刚体从视口顶部附近出发，必须在最短时间内飞向并稳定停靠在画面中央的目标平台上，且尽可能少地使用引擎推力。  
主要目标：**到达目标位置并安全、稳定地着陆**（接近目标、减小速度、保持直立姿态、双腿接触平台）。  
次要目标：耗费更少的燃料（即减少不必要的引擎动作），并尽可能快地完成着陆。  
不该混淆的目标：纯粹的速度最小化或仅追求时间最短而不考虑着陆稳定性。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（推断）
- obs[0]: x_position – 相对于目标平台的水平坐标，reward_usable: true
- obs[1]: y_position – 相对于目标平台高度的垂直坐标，reward_usable: true
- obs[2]: x_velocity – 水平线速度，reward_usable: true
- obs[3]: y_velocity – 垂直线速度，reward_usable: true
- obs[4]: body_angle – 机体倾斜角度，reward_usable: true
- obs[5]: angular_velocity – 角速度，reward_usable: true
- obs[6]: left_support_contact – 左支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true
- obs[7]: right_support_contact – 右支撑腿接触标志（1.0 接触，0.0 未接触），reward_usable: true

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- action/action_dim 0: no_engine（无任何引擎推力）
- action/action_dim 1: left_orientation_engine（启动左侧姿态引擎，施加旋转冲量）
- action/action_dim 2: main_engine（启动主引擎，产生纵向推力）
- action/action_dim 3: right_orientation_engine（启动右侧姿态引擎，施加反向旋转冲量）

## 5. step 与终止条件分析
### 5.1 终止模式
- **success-like termination:** `body_not_awake_or_settled`（机体停止运动/休眠，通常意味着已经稳定着陆在目标平台上）。
- **failure-like termination:** `crash_or_body_contact`（发生碰撞或除双腿外其他部位接触地面），`horizontal_position_outside_viewport`（水平位置飞出视口边界）。
- **ambiguous termination:** 无。
- **truncation:** 源码未显式提供截断，但环境可能带有默认的最大回合步数，其信息未在给出的 step 源码中体现，暂视为不存在。

### 5.2 success/failure 信号可用性
- explicit_success_flag_available: false
- explicit_failure_flag_available: false
- allowed_info_fields: 当前 step 源码返回空字典 `{}`
- forbidden_or_uncertain_info_fields: 所有 info 字段均不可用（不存在）

## 7. 可用于奖励函数的信号
- **position:** x_position, y_position（均为相对于目标的坐标，接近零时表示已到达）
- **velocity:** x_velocity, y_velocity（用于抑制速度，特别是在着陆阶段）
- **orientation:** body_angle, angular_velocity（用于抑制滚动与倾斜）
- **contact:** left_support_contact, right_support_contact（标志是否已安全落在平台上）
- **action/engine:** action index（可区分是否启动引擎、哪一个引擎，以便进行燃料惩罚）
- **other:** 可从组合状态中提取“着陆成功”的复合特征（位置近零、速度极小、角度极小、双腿均接触）

# 历史记忆
# Reward Memory

| iter | skeleton | score | best | delta | len | key_signal | action |
|---:|---|---:|---:|---:|---:|---|---|
| 1 | engine_efficiency + orientation_penalty + progress_to_goal + successful_settle_proxy | -121.66 | -121.66 | 0.00 | 68.35 | engine_efficiency=-0.014 orientation_penalty=-0.020 progress_to_goal=-0.972 successful_settle_proxy=0.035 | new_best |
| 2 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + settle_reward + velocity_damping | 108.12 | 108.12 | 0.00 | 743.30 | distance_penalty=-0.127 engine_penalty=-0.033 orientation_penalty=-0.004 progress_delta=0.003 settle_reward=2.290 | new_best |
| 3 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + settle_delta + velocity_damping | -121.20 | 108.12 | -229.31 | 68.40 | distance_penalty=-0.486 engine_penalty=-0.009 orientation_penalty=-0.008 progress_delta=0.049 settle_delta=0.091 | no_meaningful_improvement |
| 4 | distance_penalty + engine_penalty + landing_bonus + orientation_penalty + progress_delta + settle_reward | 14.06 | 108.12 | -94.06 | 264.10 | distance_penalty=-0.148 engine_penalty=-0.021 landing_bonus=3.551 orientation_penalty=-0.002 progress_delta=0.077 | no_meaningful_improvement |
