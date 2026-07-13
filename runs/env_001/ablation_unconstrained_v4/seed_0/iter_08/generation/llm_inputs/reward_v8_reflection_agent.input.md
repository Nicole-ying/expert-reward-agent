# Search objective
- target_score: 200.000000
- current_score: 16.493080
- gap_to_target: 183.506920
- target_achievement_ratio: 8.247%
Use the target only to judge search progress. Do not reverse-engineer or reproduce an official reward formula.

# 上一轮奖励函数代码（该轮得分: 16.493080）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Splits iter 2's exploit-prone settle_reward into two-layer design:
    - proximity_reward: smooth distance-only signal (100% active, no cliff)
    - stable_landing_bonus: proximity-gated stability bonus (soft factors, extra credit)
    Prevents both the hover exploit (iter 2) and the sparse cliff (iter 5).
    """
    px, py, pvx, pvy, pangle, pang_vel, pleft_contact, pright_contact = obs
    nx, ny, nvx, nvy, nangle, nang_vel, nleft_contact, nright_contact = next_obs

    # --- Distances ---
    prev_distance = (px ** 2 + py ** 2) ** 0.5
    curr_distance = (nx ** 2 + ny ** 2) ** 0.5

    # --- Speed ---
    curr_speed = (nvx ** 2 + nvy ** 2) ** 0.5

    # --- Proximity gate (continuous, always active) ---
    k_prox = 7.0
    curr_prox = 1.0 / (1.0 + k_prox * curr_distance)

    # ============================================================
    # Component A: progress_delta (primary approach driver)
    # ============================================================
    w_progress = 15.0
    progress_reward = w_progress * (prev_distance - curr_distance)

    # ============================================================
    # Component B: distance_penalty (global baseline pull toward origin)
    # ============================================================
    w_dist = 0.3
    distance_penalty = -w_dist * curr_distance

    # ============================================================
    # Component C: proximity_reward (smooth, always-active approach incentive)
    # Replaces iter 2's settle_reward pure-proximity base.
    # 100% active, no threshold cliff: provides continuous gradient toward target.
    # ============================================================
    w_prox = 3.0
    proximity_reward = w_prox * curr_prox

    # ============================================================
    # Component D: stable_landing_bonus (proximity-gated stability extra)
    # Only meaningful near target; soft factors avoid hard-threshold cliff.
    # Multiplies proximity so the bonus scales with closeness.
    # ============================================================
    speed_ok = max(0.0, 1.0 - curr_speed / 0.5)
    angle_ok = max(0.0, 1.0 - abs(nangle) / 0.2)
    contact_factor = 0.5 * (nleft_contact + nright_contact)
    w_stable = 4.0
    stable_landing_bonus = w_stable * curr_prox * speed_ok * angle_ok * (1.0 + contact_factor)

    # ============================================================
    # Component E: velocity_damping (global mild speed penalty, prox-gated)
    # ============================================================
    w_vel = 0.3
    velocity_damping = -w_vel * curr_prox * curr_speed

    # ============================================================
    # Component F: orientation_penalty (proximity-gated)
    # ============================================================
    w_orient = 0.3
    w_angvel = 0.1
    orientation_penalty = -w_orient * curr_prox * (nangle ** 2 + w_angvel * nang_vel ** 2)

    # ============================================================
    # Component G: engine_penalty (fuel efficiency)
    # ============================================================
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # --- Total ---
    total_reward = (
        progress_reward
        + distance_penalty
        + proximity_reward
        + stable_landing_bonus
        + velocity_damping
        + orientation_penalty
        + engine_penalty
    )

    reward_components = {
        "progress_delta": float(progress_reward),
        "distance_penalty": float(distance_penalty),
        "proximity_reward": float(proximity_reward),
        "stable_landing_bonus": float(stable_landing_bonus),
        "velocity_damping": float(velocity_damping),
        "orientation_penalty": float(orientation_penalty),
        "engine_penalty": float(engine_penalty),
    }

    return float(total_reward), reward_components
```

# 训练反馈（上一轮代码的训练结果）
# Training Feedback

## Final-policy outcome
score=16.493080, len=1000.000000, terminated=0/20, truncated=20/20, reward_errors=0
score_range=[-34.957828, 92.668232]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| stable_landing_bonus | 1649.524493 | 51.8% | 51.8% | 96.5% |
| proximity_reward | 1384.676214 | 43.5% | 43.5% | 100.0% |
| distance_penalty | -88.176930 | -2.8% | 2.8% | 100.0% |
| engine_penalty | -29.187000 | -0.9% | 0.9% | 97.3% |
| progress_delta | 20.170868 | 0.6% | 0.7% | 100.0% |
| velocity_damping | -8.972703 | -0.3% | 0.3% | 100.0% |
| orientation_penalty | -0.227809 | -0.0% | 0.0% | 100.0% |

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
| 5 | distance_penalty + engine_penalty + landing_reward + orientation_penalty + progress_delta + velocity_damping | -117.75 | 108.12 | -225.86 | 68.35 | distance_penalty=-0.292 engine_penalty=-0.006 landing_reward=0.028 orientation_penalty=-0.003 progress_delta=0.194 | unsolved_high_achievement_continue_from_best |
| 6 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + proximity_reward + stable_landing_bonus | 16.49 | 108.12 | -91.62 | 1000.00 | distance_penalty=-0.130 engine_penalty=-0.023 orientation_penalty=-0.008 progress_delta=0.054 proximity_reward=1.242 | no_meaningful_improvement |
| 7 | distance_penalty + engine_penalty + orientation_penalty + progress_delta + proximity_reward + stable_landing_bonus | 16.49 | 108.12 | -91.62 | 1000.00 | distance_penalty=-0.130 engine_penalty=-0.023 orientation_penalty=-0.008 progress_delta=0.054 proximity_reward=1.242 | no_meaningful_improvement |
