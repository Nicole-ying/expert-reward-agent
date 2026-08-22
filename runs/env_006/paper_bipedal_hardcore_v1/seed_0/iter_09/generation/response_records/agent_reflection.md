# Response Record

# 设计理由

## 第 0 步：信号覆盖审计

### 0.1 终止模式分析
20 个评估 episode 全部以 `terminated` 结束（无 `truncated`），平均 episode 长度仅 230.8 步（中等偏短）。结合环境终止条件（摔倒 `body_fallen_over` 与到达终点 `reached_end_of_terrain`），推断 agent 主要因摔倒而终止，因为：
- 若主要由到达终点触发，累积前进奖励应较高，但总 score 为 **-50.6**，近乎纯负；
- 即使 gate 保持姿态稳定，短 episode 长度说明 agent 仍频繁摔倒。

### 0.2 观测使用扫描
当前代码只使用了：
- `obs[0]`（hull_angle）→ 姿态 gate
- `obs[1]`（hull_ang_vel）→ 角速度 gate
- `obs[2]`（horizontal_speed）→ 前向奖励
- `obs[3]`（vertical_speed）→ 提取但**未使用**

**完全未使用的观测**：
- 关节角度/角速度（`obs[4:12]`）——可用于步态协调
- 触地信号（`obs[12], obs[13]`）——表示支撑相，可激励交替步态
- 全部 10 个 LIDAR 测距值（`obs[14:24]`）——提供前方地形高度预览，对崎岖地形至关重要

### 0.3 信号缺口判断
- **信号缺失**——触地信号和 LIDAR 全部闲置。agent 无法感知地形或步态切换，导致在遇到坑洞/树桩时无预判能力，步态不协调，极易摔倒。
- 虽然稳定性 gate 活跃，但地形感知缺失是当前弱终止的主要候选原因。历史最佳（iter5）使用了 `contact_transition_reward`，长度达到 406.5 步，明显受益于步态激励。

### 0.4 僵尸组件检查
所有组件 `active_rate > 97%`，无僵尸组件。`energy_penalty` 贡献极小（－0.4%），可保留不动。

## 行为诊断
agent 在缺乏步态协调和地形感知的情况下，仅由姿态 gate 抑制严重倾斜，但遇到崎岖地形时步态不稳定，容易摔倒，前进距离受限。引入触地切换奖励，提供步态内部协调信号，使双腿交替支撑更自然，有望延长存活时间。

- **干预目标**：增加步态稳定性，延长 episode 长度。
- **方向**：历史 iter5 已经证明 `contact_transition_reward` 有效，应复用该信号。
- **层级**：Level 2——新增组件（利用未使用的触地观测）。

## 修改内容
在原有 `forward_reward * stability_gate - energy_penalty` 基础上，**新增 `contact_transition_reward`**：
- 从 `obs[12], obs[13]` 和 `next_obs[12], next_obs[13]` 提取触地状态；
- 计算两腿触地状态的绝对变化量，奖励任何触地/离地事件（鼓励交替步态）；
- 系数 0.05 使每步贡献远小于前向主信号（0.05 × 1.5 事件 ≈ 0.075 per-step，主信号 per-step ≈ 0.37，满足 ≤0.3x 的惩罚约束，此处为奖励，更无负担问题）。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # =========================================================================
    # 1. 提取观测信号
    # =========================================================================
    hull_angle = obs[0]
    hull_ang_vel = obs[1]
    horizontal_speed = obs[2]
    leg_1_contact = obs[12]
    leg_2_contact = obs[13]

    next_hull_angle = next_obs[0]
    next_hull_ang_vel = next_obs[1]
    next_horizontal_speed = next_obs[2]
    next_leg_1_contact = next_obs[12]
    next_leg_2_contact = next_obs[13]

    # =========================================================================
    # 2. 前向速度奖励 (主学习信号)
    # =========================================================================
    forward_speed = max(0.0, horizontal_speed)
    forward_reward = 1.0 * forward_speed

    # =========================================================================
    # 3. 姿态稳定门
    # =========================================================================
    tilt_safe_bound = 0.3
    tilt_danger_bound = 0.7
    tilt_margin = tilt_danger_bound - tilt_safe_bound

    abs_tilt = abs(hull_angle)
    if abs_tilt <= tilt_safe_bound:
        tilt_gate = 1.0
    elif abs_tilt >= tilt_danger_bound:
        tilt_gate = 0.0
    else:
        tilt_gate = 1.0 - (abs_tilt - tilt_safe_bound) / tilt_margin

    ang_vel_thresh = 2.0
    ang_vel_margin = 4.0
    abs_ang_vel = abs(hull_ang_vel)
    if abs_ang_vel <= ang_vel_thresh:
        ang_vel_factor = 1.0
    elif abs_ang_vel >= ang_vel_thresh + ang_vel_margin:
        ang_vel_factor = 0.3
    else:
        ang_vel_factor = 1.0 - 0.7 * (abs_ang_vel - ang_vel_thresh) / ang_vel_margin

    stability_gate = tilt_gate * ang_vel_factor

    # =========================================================================
    # 4. 接触切换奖励 (新增 — 激励交替步态)
    # =========================================================================
    leg1_change = abs(next_leg_1_contact - leg_1_contact)
    leg2_change = abs(next_leg_2_contact - leg_2_contact)
    contact_transition_reward = 0.05 * (leg1_change + leg2_change)

    # =========================================================================
    # 5. 能量效率惩罚 (轻量)
    # =========================================================================
    action_sq_sum = action[0]**2 + action[1]**2 + action[2]**2 + action[3]**2
    energy_penalty = 0.005 * action_sq_sum

    # =========================================================================
    # 6. 组合并返回
    # =========================================================================
    gated_forward = forward_reward * stability_gate
    total_reward = gated_forward + contact_transition_reward - energy_penalty

    components = {
        "forward_reward": forward_reward,
        "stability_gate": stability_gate,
        "gated_forward": gated_forward,
        "contact_transition_reward": contact_transition_reward,
        "energy_penalty": -energy_penalty
    }

    return float(total_reward), components
```

# 诊断摘要
- **audit**: 触地信号与 LIDAR 完全未用，步态缺乏内部激励是短 episode 的主因；历史最佳已证明接触奖励有效。
- **behavior**: agent 以姿态 gate 维持平衡，但因无步态协调与地形预判，频繁摔倒，平均存活仅 230 步。
- **signal**: 缺少步态切换（触地事件）信号，接触维度的信息未被转化为奖励。
- **level**: Level 2
- **hypothesis**: 添加接触切换奖励后，agent 将学会有节奏的交替双支撑步态，改善崎岖地形的稳定性，延长存活时间，提升总 forward 累积和 score。
- **risk**: 若切换奖励过强可能导致 agent 原地高频踏步以刷分，但前向速度 gate 会压制零前进速度的踏步，实现自然平衡。
