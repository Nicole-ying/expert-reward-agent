# 1. Search objective
- target_score: 200.000000
- current_score: -11.598658
- gap_to_target: 211.598658
- target_achievement_ratio: -5.799%

# 2. 上一轮奖励函数代码（该轮得分: -11.598658）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 解包观测
    x_next, y_next = next_obs[0], next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---- 超参数 ----
    PROGRESS_WEIGHT = 20.0
    SURVIVAL_PENALTY = -0.08
    FAIL_PENALTY = -30.0               # 出界 / 坠毁 / 远距离静止 一次性惩罚

    ANGLE_PENALTY = 0.3
    ANG_VEL_PENALTY = 0.03

    ACTION_FUEL_PENALTY = -0.01

    # 成功着陆检测参数（收紧以杜绝 exploit）
    SUCCESS_DIST_THRESH = 0.15
    SUCCESS_SPEED_THRESH = 0.2
    SUCCESS_ANGLE_THRESH = 0.1
    LANDING_SUCCESS_BONUS = 150.0

    # 出界/坠毁阈值
    X_BOUNDARY = 1.0
    GROUND_Y_CLOSE = 0.15
    CRASH_ANGLE = 0.8
    CRASH_IMPACT_VEL = 1.5

    # ---- 1. 进展信号 ----
    x_cur, y_cur = obs[0], obs[1]
    dist_cur = (x_cur ** 2 + y_cur ** 2) ** 0.5
    dist_next = (x_next ** 2 + y_next ** 2) ** 0.5
    progress = PROGRESS_WEIGHT * (dist_cur - dist_next)

    # ---- 2. 每步存在惩罚 ----
    survival = SURVIVAL_PENALTY

    # ---- 3. 姿态/稳定惩罚 ----
    stability = -ANGLE_PENALTY * (body_angle_next ** 2) - ANG_VEL_PENALTY * (ang_vel_next ** 2)

    # ---- 4. 燃料效率惩罚 ----
    fuel = ACTION_FUEL_PENALTY if action != 0 else 0.0

    # ---- 5. 一次性成功着陆奖励（收紧阈值杜绝反复触发） ----
    # 上一帧状态
    prev_dist = dist_cur
    prev_speed = (obs[2] ** 2 + obs[3] ** 2) ** 0.5
    prev_angle = abs(obs[4])
    prev_contact = max(obs[6], obs[7]) > 0.5

    # 当前帧状态
    cur_speed = (x_vel_next ** 2 + y_vel_next ** 2) ** 0.5
    cur_angle = abs(body_angle_next)
    cur_contact = max(left_contact, right_contact) > 0.5

    # 判断成功条件（收紧）
    prev_success = (prev_dist < SUCCESS_DIST_THRESH and prev_speed < SUCCESS_SPEED_THRESH and
                    prev_angle < SUCCESS_ANGLE_THRESH and prev_contact)
    cur_success = (dist_next < SUCCESS_DIST_THRESH and cur_speed < SUCCESS_SPEED_THRESH and
                   cur_angle < SUCCESS_ANGLE_THRESH and cur_contact)

    # 仅在首次满足时授予
    landing_success_bonus = LANDING_SUCCESS_BONUS if (cur_success and not prev_success) else 0.0

    # ---- 6. 终止事件推断（失败检测） ----
    out_of_bounds = abs(x_next) > X_BOUNDARY

    crash = False
    if (left_contact > 0.5 or right_contact > 0.5):
        close_to_ground = y_next < GROUND_Y_CLOSE
        excessive_tilt = abs(body_angle_next) > CRASH_ANGLE
        high_impact = abs(y_vel_next) > CRASH_IMPACT_VEL
        if close_to_ground and (excessive_tilt or high_impact):
            crash = True

    # 远距离静止失败
    distant_stall = (
        (y_next < GROUND_Y_CLOSE) and
        (dist_next > SUCCESS_DIST_THRESH) and
        cur_speed < SUCCESS_SPEED_THRESH and
        cur_angle < SUCCESS_ANGLE_THRESH and
        (not (left_contact > 0.5 or right_contact > 0.5))
    )

    failure_penalty = FAIL_PENALTY if (out_of_bounds or crash or distant_stall) else 0.0

    # 合并奖励
    total_reward = (progress + survival + stability + fuel +
                    landing_success_bonus + failure_penalty)

    components = {
        'progress': progress,
        'survival': survival,
        'stability': stability,
        'fuel': fuel,
        'landing_success_bonus': landing_success_bonus,
        'failure_penalty': failure_penalty
    }

    return float(total_reward), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 997.95 | -166.96 | ✅ |
| 2 | 通过 delta-distance + survival penalty + terminal success b... | 通过 delta-distance + survival penalty + terminal success b... | 109.45 | 14.03 | ✅ |
| 3 | 加入 early‑stop 惩罚后，agent 在远离目标时必须保持移动，不能在远处“赖着”终止，从而被迫前往目标... | 加入 early‑stop 惩罚后，agent 在远离目标时必须保持移动，不能在远处“赖着”终止，从而被迫前往目标... | 74.25 | -39.65 | ❌ |
| 4 | 移除 early_stop 后，agent 恢复以 progress（距离缩短）为主导的动机，回归 Iter2 的... | 移除 early_stop 后，agent 恢复以 progress（距离缩短）为主导的动机，回归 Iter2 的... | 109.45 | 14.03 | ❌ |
| 5 | 加入远离目标的静止失败惩罚后，agent 将不能再安全地结束于远处，必须努力到达目标垫，从而提升成功率和平均得分。 | 加入远离目标的静止失败惩罚后，agent 将不能再安全地结束于远处，必须努力到达目标垫，从而提升成功率和平均得分。 | 127.10 | 12.25 | ➖ |
| 6 | 连续着陆激励会渐进强化接近完美着陆的所有子目标，驱动 agent 在最后阶段主动减速、调平、触地，显著提升成功率与得分。 | 连续着陆激励会渐进强化接近完美着陆的所有子目标，驱动 agent 在最后阶段主动减速、调平、触地，显著提升成功率与得分。 | 71.25 | -60.09 | ❌ |
| 7 | 骨架变化: failure_penalty + fuel + landing_success_bonus + p | — | 579.05 | 164.34 | ✅ |
| 8 | 骨架变化: failure_penalty + fuel + landing_success_bonus + p | — | 80.75 | -11.60 | ❌ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=-11.598658, len=80.750000, terminated=20/20, truncated=0/20, reward_errors=0
score_range=[-71.296190, 49.223280]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_success_bonus | 30.000000 | 49.6% | 49.6% | 0.2% |
| progress | 22.441013 | 37.1% | 38.3% | 100.0% |
| survival | -6.460000 | -10.7% | 10.7% | 100.0% |
| stability | -0.471286 | -0.8% | 0.8% | 100.0% |
| fuel | -0.392000 | -0.6% | 0.6% | 48.5% |
| failure_penalty | 0.000000 | 0.0% | 0.0% | 0.0% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 3/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 6. 环境事实（只据此理解任务和变量，不猜测环境名称）
## 1. 任务目标
本环境是一个 2D 轨迹优化与精准着陆任务。一架拥有主引擎和双定向引擎的飞行器从视口顶部中心附近随机初始力释放，需要尽快飞到中心目标垫（着陆平台）上，实现安全、稳定的接触着陆。主要目标是**快速到达并悬停/降落在目标垫上**，次要目标是**节约燃料（减少引擎使用）**，同时在接近过程中保持姿态稳定，避免猛烈碰撞或飞出视口。任务不应被误解为持续前进或单纯生存平衡：它是以到达指定位置并“软接触”为核心的导航目标到达任务。

## 3. 观察空间 observation_space
- type: Box
- shape: (8,)
- dtype: float32（根据典型实现推断）
- **obs[0]**: `x_position` – 飞行器相对于目标垫的水平坐标（偏移），reward_usable: true （可用作距离度量）
- **obs[1]**: `y_position` – 飞行器相对于垫高度的垂直坐标（偏移），reward_usable: true （距离度量）
- **obs[2]**: `x_velocity` – 水平线速度，reward_usable: true （可惩罚/奖励减速）
- **obs[3]**: `y_velocity` – 垂直线速度，reward_usable: true （同上）
- **obs[4]**: `body_angle` – 机体倾角，reward_usable: true （可鼓励保持水平）
- **obs[5]**: `angular_velocity` – 角速度，reward_usable: true （可惩罚过快的旋转）
- **obs[6]**: `left_support_contact` – 左侧支撑脚接触标志（0 或 1），reward_usable: true （可用于判断着陆状态）
- **obs[7]**: `right_support_contact` – 右侧支撑脚接触标志，reward_usable: true （同上）

## 4. 动作空间 action_space
- type: Discrete
- n: 4
- **action 0**: `no_engine` – 不点火，飞行器仅受重力、风等影响（节省燃料的动作）
- **action 1**: `left_orientation_engine` – 启动左侧定向引擎，产生力矩调整姿态
- **action 2**: `main_engine` – 启动主引擎，产生向上的推力（可能也有水平分量，取决于姿态）
- **action 3**: `right_orientation_engine` – 启动右侧定向引擎，反向力矩调整姿态

## 5. step 与终止条件分析

### 5.1 终止模式
- **success-like termination**: 飞行器在目标垫附近稳定停靠（触发 `body_not_awake_or_settled`，且位置靠近原点，速度极小，倾角接近 0）。环境未显式提供成功 flag，需从观测组合推断。
- **failure-like termination**: 
  - `horizontal_position_outside_viewport`：飞行器飞出视口水平边界→失败。
  - `crash_or_body_contact`（但与目标垫安全接触不同的碰撞）：若飞行器身体（非支撑脚）触地或触垫，或姿态严重倾覆后接触→失败。
- **ambiguous termination**: 在远离目标处触发 `body_not_awake_or_settled`（如早期静止在顶部或其他位置）→应视为失败或无效终止。
- **truncation**: 未在 masked step 中出现，可能无时间截断，但实际环境通常有最大步数（未说明，此处只考虑给定的终止条件）。

### 5.2 success/failure 信号可用性
- **explicit_success_flag_available**: false
- **explicit_failure_flag_available**: false
- **allowed_info_fields**: 没有任何明确可用的字段（step 返回 `{}`）
- **forbidden_or_uncertain_info_fields**: 不允许依赖 info 中的任何字段；如环境中存在的“success”、“failure”等均为未知，不能直接使用。

**推断路径**：通过观测信号间接区分成功/失败。成功着陆特征：`terminated=True` 时 `|x_position| < ε_x`、`|y_position| < ε_y`（接近垫中心）、`|x_velocity|, |y_velocity|` 接近 0、`|body_angle|` 接近 0、`left_support_contact` 与 `right_support_contact` 至少一个为 1（或两者为 1）。失败则表现为出界、远离目标时的静止、或高速/大角度接触。这些均属于 derived_possible 信号。

## 7. 可用于奖励函数的信号
从观测与终止推断的角度：
- **position**:
  - `x_position`（obs[0]）、`y_position`（obs[1]）— 反映到目标垫的欧氏距离或分量距离。
  - 可从 `next_obs` 获取下一步位置。
- **velocity**:
  - `x_velocity`（obs[2]）、`y_velocity`（obs[3]）— 反映接近速度，可用于减速奖励。
- **orientation**:
  - `body_angle`（obs[4]）— 应保持接近 0（水平）。
  - `angular_velocity`（obs[5]）— 抑制过快自旋。
- **contact**:
  - `left_support_contact`（obs[6]）、`right_support_contact`（obs[7]）— 脚部与垫接触，可用于终端着陆检测或稳定性奖励。
- **action/engine**:
  - `action` 本身可用于惩罚引擎使用（燃料惩罚），三个有推力动作为 1,2,3，可赋予不同权重。
- **other**:
  - 终端事件推断：根据终止时的 `next_obs` 状态组合判断成功着陆或失败（出界、坠毁等），提供 derived_possible 奖励/惩罚。

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
| 1 | failure_penalty + landing_proxy + progress + stability_penalty | -166.96 | -166.96 | 0.00 | 997.95 | failure_penalty=-0.005 landing_proxy=0.764 progress=0.001 stability_penalty=-0.008 | new_best |
| 2 | failure_penalty + fuel + landing_bonus + progress + stability + survival | 14.03 | 14.03 | 0.00 | 109.45 | failure_penalty=-0.022 fuel=-0.007 landing_bonus=0.762 progress=0.275 stability=-0.015 | new_best |
| 3 | failure_penalty + fuel + landing_bonus + progress + stability + survival | -39.65 | 14.03 | -53.68 | 74.25 | failure_penalty=-0.035 fuel=-0.006 landing_bonus=0.658 progress=0.293 stability=-0.011 | no_meaningful_improvement |
| 4 | failure_penalty + fuel + landing_bonus + progress + stability + survival | 14.03 | 14.03 | 0.00 | 109.45 | failure_penalty=-0.022 fuel=-0.007 landing_bonus=0.762 progress=0.275 stability=-0.015 | no_meaningful_improvement |
| 5 | failure_penalty + fuel + landing_bonus + progress + stability + survival | 12.25 | 14.03 | -1.78 | 127.10 | failure_penalty=-0.035 fuel=-0.008 landing_bonus=0.766 progress=0.260 stability=-0.019 | unsolved_high_achievement_continue_from_best |
| 6 | failure_penalty + fuel + landing_prox + progress + stability + survival | -60.09 | 14.03 | -74.11 | 71.25 | failure_penalty=-0.030 fuel=-0.005 landing_prox=0.008 progress=0.303 stability=-0.009 | no_meaningful_improvement |
| 7 | failure_penalty + fuel + landing_success_bonus + progress + stability + survival | 164.34 | 164.34 | 0.00 | 579.05 | failure_penalty=-1.239 fuel=-0.008 landing_success_bonus=0.558 progress=0.098 stability=-0.010 | new_best |
| 8 | failure_penalty + fuel + landing_success_bonus + progress + stability + survival | -11.60 | 164.34 | -175.94 | 80.75 | failure_penalty=-0.043 fuel=-0.007 landing_success_bonus=0.398 progress=0.296 stability=-0.014 | no_meaningful_improvement |
