# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    v1 reward: progress guided by preview of upcoming terrain roughness,
    with a hinge penalty on excessive torso tilt.
    """
    # --- Extract observations ------------------------------------------------
    hull_angle = obs[0]                # torso tilt (rad)
    horizontal_speed = obs[2]          # forward speed (m/s)
    lidar = obs[14:24]                 # 10 terrain height readings

    # --- Component A: forward progress ---------------------------------------
    # Encourage forward motion; ignored when moving backwards.
    progress_raw = max(0.0, horizontal_speed)
    w_progress = 1.0

    # --- Component B: preview factor from lidar roughness --------------------
    # Higher terrain roughness -> reduce effective progress, encouraging
    # the agent to slow down / adjust gait before obstacles.
    n_lid = len(lidar)
    if n_lid > 0:
        mean_l = sum(lidar) / n_lid
        # variance of lidar readings as roughness measure
        var_l = sum((l - mean_l) ** 2 for l in lidar) / n_lid
        roughness = var_l ** 0.5
    else:
        roughness = 0.0

    # preview_factor in [preview_min, 1.0]; 1.0 on flat ground, decays with roughness
    k_preview = 2.0          # sensitivity to roughness
    preview_factor = 1.0 / (1.0 + k_preview * roughness)

    progress_reward = w_progress * progress_raw * preview_factor

    # --- Component C: posture hinge penalty ----------------------------------
    # Penalize dangerous torso tilt beyond a safe threshold.
    torso_threshold = 0.5          # ~28.6 degrees
    w_posture = 1.0
    excess_tilt = max(0.0, abs(hull_angle) - torso_threshold)
    posture_penalty = -w_posture * excess_tilt

    # --- Combine -------------------------------------------------------------
    total_reward = progress_reward + posture_penalty
    components = {
        "progress_reward": progress_reward,
        "posture_penalty": posture_penalty
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- **task_family**: locomotion_continuous_control
- **dynamics_subtype**: planar_bipedal_gait on rough terrain

## selected reward roles 与 role_to_signal_mapping

| role | signal | obs index | formula operator |
|---|---|---|---|
| 主学习信号 (前进进度) | horizontal_speed | 2 | `progress_raw = max(0, speed)`; 然后乘 preview_factor（见下） |
| 预览调节 (前方地形崎岖度) | lidar_1..lidar_10 | 14‑23 | roughness = lidar标准差；preview_factor = 1/(1+k*roughness)，乘到 progress 上 |
| 稳定/安全约束 (躯干倾斜) | hull_angle | 0 | hinge‑penalty `max(0, abs(angle)-threshold)` |

## component 细节

- **progress_reward**: 直接鼓励向前运动，使用非负水平速度作为基础信号。系数 `w_progress=1.0`。
- **preview_factor**: 利用 LIDAR 读数计算前方地形高度的标准差，代表地形崎岖程度。崎岖度高时 preview_factor 衰减，降低 progress_reward，引导 agent 在接近障碍时主动减速、调整步态，减少摔倒。这是本设计与历史尝试的主要 **差异化** 点——历史方案均未使用 LIDAR 信息。
- **posture_penalty**: 采用 hinge 形式，仅当躯干倾斜超过 0.5 rad 时才施加线性惩罚，避免不必要的全时压制。`w_posture=1.0` 保证倾斜超限时惩罚显著，但不至于完全抵消前进奖励。

## excluded roles 及原因

| role | 原因 |
|---|---|
| terminal_success_reward | 环境未提供显式成功标志 (`explicit_success_flag_available`=false) |
| terminal_failure_penalty | 无明确的失败终止标志，且从观测可靠推断终止状态困难 (`explicit_failure_flag_available`=false) |
| action_efficiency / action_cost | 留到后续迭代；v1 先让 agent 学习稳定前进，再优化能耗 |
| soft_health_gate | 已被 preview_factor 和 posture hinge 覆盖，暂不引入额外 gate 避免过早限制探索 |
| air_penalty / vertical_speed_penalty / angular_velocity_penalty | 历史实验表明这些组件效果不佳，本次设计以地形预览为核心，舍去效果不明确的约束项 |

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境未声明存在显式的成功/失败标志，且 `info` 字段为空。观察空间虽有地面接触与角度信息，但难以可靠地区分“摔倒终止”与“正常时间结束”，硬写 terminal_event 易引入虚假信号，故放弃。

## 留到后续迭代的职责
- **动作效率** (action_efficiency)：在 agent 学会基本行走后，加入小权重的力矩惩罚以节省能量。
- **动态课程 / 自适应权重**：若出现“短板”阶段（如前期总摔倒、后期力矩过大），可基于 training_progress 调整组件权重。
- **更精细的摔倒预警**：可结合 leg_contact、垂直速度等构造复合摔倒前兆信号，用于附加救助惩罚或 gate。

## 训练后应观察的 failure modes
- **站立不动**：若 speed 几乎为零且 progress_reward 被 preview_factor 压低，agent 可能选择静止。观察水平速度分布。
- **剧烈摇摆但未摔倒**：posture_penalty 的 hinge 阈值可能设得过宽松，导致 agent 在危险边缘反复振荡。需监控 hull_angle 的 episode 均值与最大值。
- **地形预览失效**：若 LIDAR 读数方差不能有效区分平坦与崎岖地段，preview_factor 将退化为常数 1，失去调节作用。应观察 roughness 分布的方差，必要时调整 k_preview 或采用更稳健的粗糙度度量。
- **负奖励持续**：若 total reward 仍长时间为负，考虑适当提高 progress 权重或降低 preview 衰减灵敏度，避免 terrain gate 过度抑制前进动机。
