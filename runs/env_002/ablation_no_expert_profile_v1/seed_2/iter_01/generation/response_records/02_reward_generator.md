# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Primary progress: reward forward horizontal velocity (only positive values)
    forward_vel = next_obs[2]
    forward_reward = max(forward_vel, 0.0)

    # Stability health penalties
    hull_angle = next_obs[0]          # body tilt from vertical
    hull_ang_vel = next_obs[1]        # body angular velocity
    vertical_vel = next_obs[3]        # vertical speed (bouncing)

    stability_cost = (
        hull_angle ** 2 +
        hull_ang_vel ** 2 +
        vertical_vel ** 2
    )

    # Component weights (v1)
    w_forward = 1.0
    w_stability = 0.1

    total_reward = w_forward * forward_reward - w_stability * stability_cost

    components = {
        'forward_reward': w_forward * forward_reward,
        'stability_penalty': -w_stability * stability_cost
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- 任务族：locomotion_continuous_control  
- 动力学子类型：planar_bipedal_gait（平面双足步态前进）

## selected reward roles
1. **主学习信号（mandatory）**：前进驱动力  
   - 角色：持续鼓励 agent 向前移动，是实现“尽可能远、尽可能快”的核心。  
   - 信号：`next_obs[2]`（horizontal_velocity），每步提供稠密梯度。  
   - 公式算子：`dense_state_signal` 的正向形式，`max(signal, 0)`，只奖励正方向速度，避免对后退施加惩罚，保持梯度连贯。

2. **稳定/安全约束（conditional）**：姿态与运动平滑  
   - 角色：惩罚身体大幅倾斜、快速旋转和剧烈垂直震荡，促进稳定行走。  
   - 信号：  
     - `next_obs[0]`（hull_angle）  
     - `next_obs[1]`（hull_angular_velocity）  
     - `next_obs[3]`（vertical_velocity）  
   - 公式算子：`quadratic_penalty`，三个二次项组合成一个 `stability_cost`，整体乘以轻量系数，避免过度限制探索。

## role_to_signal_mapping
- 前进速度：`next_obs[2]` → `forward_reward`  
- 姿态约束：`next_obs[0]`, `next_obs[1]`, `next_obs[3]` → `stability_penalty`

## excluded roles 及原因
- **terminal_success_reward** / **terminal_failure_penalty**：环境未提供显式成功/失败标志（`info` 为空），不可使用。  
- **步态交替奖励**：接触信号 `leg1_contact / leg2_contact` 可使用，但 v1 阶段主前进奖励已能自然诱导出步态，过早加入交替约束可能引入 reward hacking，留到后续迭代。  
- **动作能耗代价**：附属目标包括最小化能量消耗，但 v1 优先建立前行能力，加入扭矩惩罚可能让 agent 不敢发力，相关信号保留，可在 v2 或后续版本添加极小权重。  
- **复合任务完成近似（joint_condition_proxy）**：缺乏明确的接近终点信号，且在 v1 阶段前进速度已经提供足够驱动力，无需构造 soft success。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
- `explicit_success_flag_available` 与 `explicit_failure_flag_available` 均为 false，`info` 无任何可用字段；强行使用空字典或假设终止标志会违反接口契约且无法执行。

## 哪些职责留到后续迭代
- **动作能耗优化**：可引入 `-w_energy * (action ** 2).sum()`  
- **步态节律引导**：基于接触信号的交替支撑奖励  
- **动态稳定性门控**（soft_health_gate）：若前进速度导致频繁摔倒，可在后续版本将主奖励乘以姿态阈值因子  
- **curriculum_weighting**：若训练初期探索困难、后期需微调，可引入基于 `training_progress` 的动态权重，但 v1 不启用

## 训练后应该观察哪些 failure modes
- **原地跳动**：如果稳定性惩罚欠重，agent 可能通过高频率垂直跳跃获得少量前进奖励。应检查 vertical_velocity 项是否需适度提高权重或改为 bounded 形式。  
- **速度脉冲后摔倒**：若 agent 倾向于短暂冲刺然后姿态失控，说明稳定性约束不足以遏制高风险行为，可考虑在后续引入 soft_health_gate 将前进奖励与姿态健康程度相乘。  
- **过度保守/不敢迈步**：如果稳定性惩罚过强，agent 可能选择几乎不动的策略（`forward_reward` 为零但稳定损耗也接近零）。此时需适当降低 `w_stability` 或改用 bounded 形态使约束更柔和。  
- **恒定低速行走**：若 `forward_reward` 的线性形式与稳定性惩罚形成均衡在很低速度，可能无法满足“尽可能快”的要求；后续可尝试增大 `w_forward`，或在 `forward_reward` 中引入非线性下界（如 `max(horizontal_vel - v_min, 0)`），但 v1 暂不引入。
