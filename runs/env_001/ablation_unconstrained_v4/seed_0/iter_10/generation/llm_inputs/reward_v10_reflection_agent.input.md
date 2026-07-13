# Search objective
- target_score: 200.000000
- current_score: -115.081713
- gap_to_target: 315.081713
- target_achievement_ratio: -57.541%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: -115.081713）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Fixes iter 6/7 hover exploit by:
    1. Removing proximity_reward — the always-on distance payment that incentivizes hovering
    2. Boosting progress_delta as the primary approach driver
    3. Redesigning stable_landing_bonus as a STRICT landing-only bonus:
       - Requires ALL of: near target, low speed, upright angle, both leg contacts
       - Multiplicative gating (product of independent conditions) ensures bonus ≈ 0
         unless every landing condition is simultaneously met
       - Sparse but high-value: pays meaningfully only during actual settling/landing
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (used ONLY as a modulator, never standalone) ---
    k_prox = 5.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (PRIMARY approach driver)
    # Boosted to compensate for removal of proximity_reward.
    # This must be the dominant positive signal driving the agent to the target.
    # ============================================================
    w_progress = 40.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (global baseline pull toward origin)
    # ============================================================
    w_dist = 0.5
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: stable_landing_bonus (STRICT landing-only sparse reward)
    # Multiplicative product of independent landing conditions.
    # Each factor ranges [0,1]; product is near-zero unless ALL are satisfied.
    # This replaces both proximity_reward and the old permissive bonus.
    # ============================================================
    # Independent landing-condition gates (each [0,1], 1 = perfect)
    near_target = max(0.0, 1.0 - curr_distance / 0.4)       # dist < 0.4
    speed_low = max(0.0, 1.0 - curr_speed / 0.3)            # speed < 0.3
    angle_upright = max(0.0, 1.0 - abs(nangle) / 0.25)     # |angle| < 0.25 rad
    both_contacts = nleft_contact * nright_contact           # 1.0 only if both legs touch

    w_stable = 25.0
    # Product gating: bonus > 0 only when ALL conditions are substantially met.
    # (0.5 + 0.5*both_contacts) ensures contact bonus is partial with one leg,
    # full with both — but still requires near/slow/upright to matter.
    landing_quality = near_target * speed_low * angle_upright * (0.3 + 0.7 * both_contacts)
    stable_landing_bonus = w_stable * landing_quality

    # ============================================================
    # Component D: velocity_damping (mild, proximity-gated speed penalty)
    # ============================================================
    w_vel = 0.3
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component E: orientation_penalty (mild, proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component F: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.05
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + stable_landing_bonus
        + velocity_damping
        + orientation_penalty
        + engine_penalty
    )

    components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "stable_landing_bonus": float(stable_landing_bonus),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=-115.081713, len=68.450000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-140.076175, -93.973586]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| progress_delta | 44.785259 | 50.1% | 51.8% | 100.0% |
| distance_penalty | -33.270019 | -37.2% | 37.2% | 100.0% |
| velocity_damping | -5.171897 | -5.8% | 5.8% | 100.0% |
| stable_landing_bonus | 4.178748 | 4.7% | 4.7% | 0.6% |
| engine_penalty | -0.287500 | -0.3% | 0.3% | 8.4% |
| orientation_penalty | -0.234627 | -0.3% | 0.3% | 100.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 20/20
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
| 5 | distance_penalty + engine_penalty + landing_reward + orientation_penalty + progress_delta + velocity_damping | -117.75 | 108.12 | -225.86 | 68.35 | distance_penalty=-0.292 engine_penalty=-0.006 landing_reward=0.028 orientation_penalty=-0.003 progress_delta=0.194 | unsolved_high_achievement_continue_from_best |
| 6 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + proximity_reward + stable_landing_bonus | 16.49 | 108.12 | -91.62 | 1000.00 | distance_penalty=-0.130 engine_penalty=-0.023 orientation_penalty=-0.008 progress_delta=0.054 proximity_reward=1.242 | no_meaningful_improvement |
| 7 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + proximity_reward + stable_landing_bonus | 16.49 | 108.12 | -91.62 | 1000.00 | distance_penalty=-0.130 engine_penalty=-0.023 orientation_penalty=-0.008 progress_delta=0.054 proximity_reward=1.242 | no_meaningful_improvement |
| 8 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + stable_landing_bonus + velocity_damping | -115.08 | 108.12 | -223.20 | 68.45 | distance_penalty=-0.485 engine_penalty=-0.009 orientation_penalty=-0.005 progress_delta=0.650 stable_landing_bonus=0.063 | unsolved_high_achievement_continue_from_best |
| 9 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + stable_landing_bonus + velocity_damping | -115.08 | 108.12 | -223.20 | 68.45 | distance_penalty=-0.485 engine_penalty=-0.009 orientation_penalty=-0.005 progress_delta=0.650 stable_landing_bonus=0.063 | no_meaningful_improvement |
