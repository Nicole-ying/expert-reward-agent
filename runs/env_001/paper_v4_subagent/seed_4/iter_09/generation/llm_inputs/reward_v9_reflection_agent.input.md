# 1. Search objective
- target_score: 200.000000
- current_score: 157.855989
- gap_to_target: 42.144011
- target_achievement_ratio: 78.928%

# 2. 上一轮奖励函数代码（该轮得分: 157.855989）
```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    x, y, vx, vy, angle, angvel, l_contact, r_contact = obs
    nx, ny, nvx, nvy, nangle, nangvel, nl_contact, nr_contact = next_obs

    # 1. Main progress signal: distance reduction to target pad
    dist_obs = (x**2 + y**2) ** 0.5
    dist_next = (nx**2 + ny**2) ** 0.5
    progress = dist_obs - dist_next

    # 2. Stability constraints (quadratic penalties on next state)
    angle_penalty = -0.1 * (nangle ** 2)
    angvel_penalty = -0.05 * (nangvel ** 2)

    # 3. Contact-gated soft landing attractor
    #    Only active when support legs are in contact (gate), otherwise zero.
    proximity = 2.718281828 ** (-dist_next / 0.8)
    speed_factor = max(0.0, 1.0 - (abs(nvx) + abs(nvy)) / 1.0)
    angle_factor = max(0.0, 1.0 - abs(nangle) / 0.5)
    contact_gate = float(nl_contact or nr_contact)
    landing_attractor = proximity * speed_factor * angle_factor * contact_gate

    w_progress = 10.0
    w_attractor = 1.0

    total = (w_progress * progress +
             angle_penalty + angvel_penalty +
             w_attractor * landing_attractor)

    components = {
        "progress": w_progress * progress,
        "angle_penalty": angle_penalty,
        "angvel_penalty": angvel_penalty,
        "landing_bonus": w_attractor * landing_attractor
    }

    return float(total), components
```

# 3. 累积迭代记录（本轮之前所有尝试的因果链）
| iter | 做了什么 | 预期效果 | 实际 len | 实际 score | 预判 |
|---|---:|---:|---:|---:|
| 1 | 初始生成 | — | 997.95 | -166.96 | ✅ |
| 2 | 将常驻多因子代理奖励重构为“弱接近引导 + 大额成功事件奖励”后，agent 不再有悬停激励，必须快速完成安全着陆... | 将常驻多因子代理奖励重构为“弱接近引导 + 大额成功事件奖励”后，agent 不再有悬停激励，必须快速完成安全着陆... | 455.45 | -6.02 | ✅ |
| 3 | 加入软着陆塑造因子和停滞惩罚后，agent 将被连续奖励引导快速降速、回正姿态并接近原点，同时被惩罚驱离远处静止状... | 加入软着陆塑造因子和停滞惩罚后，agent 将被连续奖励引导快速降速、回正姿态并接近原点，同时被惩罚驱离远处静止状... | 72.75 | -61.64 | ❌ |
| 4 | 放宽 crash 条件使坠毁/出界受到 -10 强信号，移除 stall 释放减速空间，agent 将学会规避危险... | 放宽 crash 条件使坠毁/出界受到 -10 强信号，移除 stall 释放减速空间，agent 将学会规避危险... | 68.40 | -124.55 | ❌ |
| 5 | 用软门控替代硬惩罚后，agent 不再因靠近边界就被立即罚死，而会收到持续、微弱的“远离危险”梯度，从而延长存活时... | 用软门控替代硬惩罚后，agent 不再因靠近边界就被立即罚死，而会收到持续、微弱的“远离危险”梯度，从而延长存活时... | 68.30 | -122.21 | ❓ |
| 6 | 骨架变化: angle_penalty + angvel_penalty + landing_bonus + p | — | 1000.00 | -90.25 | ❌ |
| 7 | 替换为指数吸引子 + 接触放大后，agent 将首次获得“靠近即得利、触地奖更丰”的密集向导，预期 episode... | 替换为指数吸引子 + 接触放大后，agent 将首次获得“靠近即得利、触地奖更丰”的密集向导，预期 episode... | 813.70 | 150.81 | ✅ |
| 8 | 接触门控使接触前仅有 `progress` 提供梯度，驱使 agent 快速抵达目标区域；接触后品质奖励激活，鼓励... | 接触门控使接触前仅有 `progress` 提供梯度，驱使 agent 快速抵达目标区域；接触后品质奖励激活，鼓励... | 933.10 | 157.86 | ✅ |

预判列连续 ≥ 3 轮 ❌ → 当前方向大概率错误，应考虑 Level 3 重建。

# 5. 本轮训练反馈
# Training Feedback

## Final-policy outcome
score=157.855989, len=933.100000, terminated=2/20, truncated=18/20, reward_errors=0
score_range=[117.465240, 274.283355]

## Final-policy reward composition

These statistics come from the same fixed evaluation episodes as `score`. Shares describe observed reward composition, not causal influence.

| component | episode_sum_mean | signed_share | magnitude_share | active_rate |
|---|---:|---:|---:|---:|
| landing_bonus | 652.471025 | 97.8% | 97.8% | 73.3% |
| progress | 13.738687 | 2.1% | 2.1% | 99.8% |
| angle_penalty | -0.330834 | -0.0% | 0.0% | 100.0% |
| angvel_penalty | -0.095876 | -0.0% | 0.0% | 66.8% |

## Evaluation distribution
- fixed_eval_seeds: 10000..10019
- early_terminal (<150 steps and score<-50): 0/20
- training_reward_errors_max: 0
- full_training_distribution_stats: component_stats.md / training_summary.json (not primary reflection evidence)


# 5.5. Subagent 调研信号（基于训练数据的自动诊断）
**Key Findings**: Mean eval reward 157.86, mean episode length 933.1, only 2/20 terminated. Landing bonus accounts for 97.8% signed share (mean 652.47). Progress is small (2.1%). Angle/angvel penalties near-zero.

**Component Anomalies**: Landing bonus dominates (>97% share). Angle and angvel penalties have negligible magnitude despite moderate active rates (100%/66.8%).

**Training Dynamics**: No temporal snapshots provided; cannot assess evolution across checkpoints.

**Signal Quality**: Angle penalty always active but contribution ~0.0%. Angvel penalty active 66.8% but near-zero. Landing bonus gated by leg contact, active 73.3%. Progress signal small. No immediate dead gates, but weak shaping outside landing.

**Evidence Confidence**: `low`

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
| 2 | failure_penalty + landing_proxy + progress + stability_penalty | -6.02 | -6.02 | 0.00 | 455.45 | failure_penalty=-0.003 landing_proxy=0.173 progress=0.026 stability_penalty=-0.016 | new_best |
| 3 | action_penalty + failure_penalty + progress + soft_landing + stability_penalty + success_reward | -61.64 | -6.02 | -55.62 | 72.75 | action_penalty=-0.021 failure_penalty=-0.007 progress=0.045 soft_landing=0.001 stability_penalty=-0.010 | no_meaningful_improvement |
| 4 | action_penalty + failure_penalty + progress + soft_landing + stability_penalty + success_reward | -124.55 | -6.02 | -118.53 | 68.40 | action_penalty=-0.011 failure_penalty=-0.219 progress=0.047 soft_landing=0.001 stability_penalty=-0.017 | no_meaningful_improvement |
| 5 | action_penalty + progress + safety_penalty + soft_landing + stability_penalty + success_reward | -122.21 | -6.02 | -116.19 | 68.30 | action_penalty=-0.009 progress=0.048 safety_penalty=0.000 soft_landing=0.001 stability_penalty=-0.014 | unsolved_stagnation_fresh_restart |
| 6 | angle_penalty + angvel_penalty + landing_bonus + progress | -90.25 | -6.02 | -84.22 | 1000.00 | angle_penalty=-0.002 angvel_penalty=-0.002 landing_bonus=6.382 progress=0.046 | no_meaningful_improvement |
| 7 | angle_penalty + angvel_penalty + landing_bonus + progress | 150.81 | 150.81 | 0.00 | 813.70 | angle_penalty=-0.003 angvel_penalty=-0.001 landing_bonus=1.415 progress=0.022 | new_best |
| 8 | angle_penalty + angvel_penalty + landing_bonus + progress | 157.86 | 157.86 | 0.00 | 933.10 | angle_penalty=-0.002 angvel_penalty=-0.001 landing_bonus=0.521 progress=0.033 | new_best |
