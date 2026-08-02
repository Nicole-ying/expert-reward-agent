# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    # obs: [x_pos, y_pos, x_vel, y_vel, body_angle, angular_vel, left_contact, right_contact]
    ox, oy, ovx, ovy, oangle, oav, olc, orc = obs
    nx, ny, nvx, nvy, nangle, nav, nlc, nrc = next_obs

    # Compute distances
    old_dist = (ox * ox + oy * oy) ** 0.5
    new_dist = (nx * nx + ny * ny) ** 0.5

    # --- Goal proximity progress (main learning signal) ---
    raw_progress = old_dist - new_dist   # positive when moving closer
    max_delta = 5.0                      # clip single-step changes
    progress = max(-max_delta, min(max_delta, raw_progress))
    progress_reward = 1.0 * progress     # weight = 1.0

    # --- Landing gentleness (constraint) ---
    CLOSE_DIST = 3.0
    SAFE_SPEED = 1.0
    LAND_WEIGHT = 0.5
    speed = (nvx * nvx + nvy * nvy) ** 0.5
    # Linear activation inside the close region (0 -> 1 as distance decreases)
    close_factor = max(0.0, 1.0 - new_dist / CLOSE_DIST)
    # Hinge penalty on excess speed, scaled by close_factor
    landing_penalty = -LAND_WEIGHT * max(0.0, speed - SAFE_SPEED) * close_factor

    # --- Orientation penalty (constraint) ---
    ANGLE_THRESHOLD = 0.3   # radians
    ORIENT_WEIGHT = 0.2
    orientation_penalty = -ORIENT_WEIGHT * max(0.0, abs(nangle) - ANGLE_THRESHOLD)

    # --- Terminal success bonus (task-completion proxy) ---
    SUCCESS_DIST = 0.2
    SUCCESS_SPEED = 0.5
    SUCCESS_ANGLE = 0.2
    SUCCESS_BONUS = 0.2
    success_bonus = 0.0
    if (new_dist < SUCCESS_DIST and speed < SUCCESS_SPEED
            and abs(nangle) < SUCCESS_ANGLE
            and (nlc > 0.5 or nrc > 0.5)):
        success_bonus = SUCCESS_BONUS

    total_reward = progress_reward + landing_penalty + orientation_penalty + success_bonus

    components = {
        "goal_proximity_progress": progress_reward,
        "landing_gentleness_penalty": landing_penalty,
        "orientation_penalty": orientation_penalty,
        "terminal_success_bonus": success_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

- **task_family**: `navigation_goal_reaching`  
- **dynamics_subtype**: `goal_approach_and_soft_contact`  
- **selected reward roles** (from `reward_role_decomposition`):  
  - `goal_proximity_progress` (mandatory) — 主学习信号，使用 **improvement_delta** 算子驱动每一步靠近目标。  
  - `landing_gentleness` (mandatory) — 安全约束，在目标区域内对过大冲击速度施加 **hinge_penalty**，避免高速撞击。  
  - `orientation_penalty` (conditional) — 稳定约束，对超出安全倾角的角向采用 **hinge_penalty**。  
  - `terminal_success_bonus` (conditional, derived_possible) — 任务完成近似信号，当位置、速度、姿态与接触条件同时高度满足时给予固定的 **sparse_bonus**。  
- **role‑to‑signal mapping**:  
  - `goal_proximity_progress` ← `obs[0：1]`, `next_obs[0：1]` → Euclidean distance delta。  
  - `landing_gentleness` ← `next_obs[2：3]` 计算速度，结合 `new_dist` 作为接近门控。  
  - `orientation_penalty` ← `next_obs[4]`（body angle）。  
  - `terminal_success_bonus` ← 多信号联合检查：`next_obs[0：5]`, `next_obs[6：7]`。  
- **formula operators used**:  
  - `improvement_delta` 用于主进展（距离减少量截断到 ±5.0）。  
  - `hinge_penalty` 用于着陆温和度和姿态约束，仅在超出安全阈值时激活。  
  - `sparse_bonus` 用于成功推断，每个满足条件的 step 提供恒常奖励，强化稳定驻留行为。  
  - 所有组件均使用显式常量权重，未引入动态门控或 curriculum。  
- **excluded roles**:  
  - `engine_efficiency_penalty`（效率代价）：v1 先让 agent 学会到达并软着陆，次优的燃料节约保留到后续迭代。  
  - `survival_only_time`、`permanent_velocity_penalty`、`full_body_contact_penalty`：与任务目标冲突或缺少可观测信号，完全排除。  
- **为何没有使用 terminal_success_reward / terminal_failure_penalty**：  
  - 环境中 `explicit_success_flag_available = false`，`info` 为空，无法从终止状态直接读取成功/失败。因此采用 **derived_possible** 的方式，在观察中构造成功条件和 proxy 奖励。  
  - 失败信号（crash、出界）未提供显式标志，且难以从观测单独准确推断，故暂未加入 hard failure penalty，以避免错误惩罚真实成功落地。  
- **留到后续迭代的职责**：  
  - 动作效率代价（`engine_efficiency_penalty`），用于减少不必要的引擎动作。  
  - 更精确的落地健康门控（如 `soft_health_gate` 联合角速度、横向漂移等），以处理振荡和侧向飞出。  
  - 动态课程或衰减权重，以平衡接近难度与安全性。  
- **训练后应观察的 failure modes**：  
  - 悬停在远离目标处：检查 progress 奖励是否饱和（delta 恒为零）或接触 reward 过早吸引 agent？  
  - 高速撞击：着陆柔和度惩罚启动过晚或权重不够，需降低 `CLOSE_DIST` 或增大 `LAND_WEIGHT`。  
  - 不敢移动：若所有约束惩罚过强导致 agent 只做空动作，需确认权重未超标。  
  - 反复横跳/振荡：角过冲过大时，`orientation_penalty` 可能需要联动角速度惩罚或调整阈值。  
  - 成功误判为 crash 惩罚：目前未加 failure penalty，避开了此风险；后续若加入需严格验证推断条件。
