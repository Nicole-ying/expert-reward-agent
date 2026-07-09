# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    假设环境观测结构:
    obs[0]: x_position_relative_to_target
    obs[1]: y_position_relative_to_pad_height
    obs[2]: x_velocity
    obs[3]: y_velocity
    obs[4]: body_angle
    obs[5]: angular_velocity
    obs[6]: left_contact_flag (0.0/1.0)
    obs[7]: right_contact_flag (0.0/1.0)
    """
    # ---- 1. Progress reward: 距离减少量 (密集主学习信号) ----
    def distance(obs_arr):
        return (obs_arr[0]**2 + obs_arr[1]**2 + 1e-8) ** 0.5

    dist_old = distance(obs)
    dist_new = distance(next_obs)
    progress_reward = dist_old - dist_new

    # ---- 2. Speed tracking reward: 期望速度引导，代替硬惩罚 ----
    # 期望速度大小与距离成线性（远快近慢），用于引导合理的飞行速度
    max_speed = 5.0          # 预设的最大期望速度
    d_ref = 1.0              # 参考距离，在此距离下期望速度达到 max_speed
    desired_speed = max_speed * min(dist_new / d_ref, 1.0)

    cur_speed = (next_obs[2]**2 + next_obs[3]**2 + 1e-8) ** 0.5
    speed_error = abs(cur_speed - desired_speed)
    lambda_speed = 0.2
    speed_tracking_reward = -lambda_speed * speed_error

    # ---- 3. Soft landing proxy: 多条件线性组合，避免乘积梯度塌缩 ----
    proximity_threshold = 0.5
    speed_threshold = 0.25
    angle_threshold = 0.2     # 弧度

    proximity_score = max(0.0, 1.0 - dist_new / proximity_threshold)
    speed_low_score = max(0.0, 1.0 - cur_speed / speed_threshold)
    angle_score = max(0.0, 1.0 - abs(next_obs[4]) / angle_threshold)
    contact_score = (next_obs[6] + next_obs[7]) * 0.5

    # 线性组合，接触条件权重略高
    soft_landing_proxy = (
        proximity_score +
        speed_low_score +
        angle_score +
        2.0 * contact_score
    )  # 最大可能值约为 5.0

    # ---- 总奖励 ----
    total_reward = progress_reward + speed_tracking_reward + soft_landing_proxy

    components = {
        'progress_reward': progress_reward,
        'speed_tracking_reward': speed_tracking_reward,
        'soft_landing_proxy': soft_landing_proxy
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## 组件角色
- **主学习信号：progress_reward**  
  距离减少量 `dist(old) - dist(new)`，每一步直接奖励向目标垫靠近的进度。这是飞行器降落的根本驱动力。

- **稳定/引导约束：speed_tracking_reward**  
  不采用传统的速度/姿态硬惩罚，而是构建一个与距离相关的期望速度场：远处允许高速，近处强制低速。实际速度与期望速度的绝对误差被惩罚，既避免“不敢动”，也避免高速撞击。通过线性函数将距离映射到期望速度，梯度清晰。

- **任务完成近似信号：soft_landing_proxy**  
  由于没有显式成功标志，通过四个连续因子（位置临近、低速、小角度、双侧接触）的**线性组合**构造一个稠密的完成信号。最大分值约 5.0，在着陆条件逐步满足时提供正反馈。抛弃之前效果不佳的乘积/几何平均形式，改为线性叠加，确保每个子条件改善都能得到有意义的梯度。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境 `info` 为空，且 `explicit_success_flag_available=false`，`explicit_failure_flag_available=false`。因此无法引入终点成功奖励或失败惩罚。

## 与之前尝试结构的关键区别
1. **主信号维持 progress_delta**，但用 `speed_tracking_reward` 替代传统的 `stability_penalty`：后者直接惩罚速度与角度，容易与进度奖励冲突；前者提供与距离自适应的速度目标，更符合“飞行－减速－着陆”的物理直觉。
2. **soft_landing_proxy** 从乘积/几何平均改为线性组合：四个因子独立贡献，避免单因子较差时奖励塌缩为零导致的“无梯度悬崖”。接触条件的权重稍高但不独占，仍需位置、速度、角度的配合才能获得高分。
3. 去除了全局稳定性惩罚，将姿态/速度的管理完全交给 `speed_tracking` 和 `soft_landing`，让系统在飞行阶段保持足够的探索自由度。

## 后续迭代留意的组件
- 效率类组件（`energy_penalty` 或 `time_penalty`）暂不引入，待 agent 能稳定完成着陆后再优化引擎使用。
- 若出现目标附近震荡，可对 `progress_reward` 进行 clipping 或引入微小的时间惩罚。
- 若 soft_landing 仍被轻触 exploit，后续可收紧阈值或增加“稳定持续”条件（但需保持线性组合）。

## 训练后应观察的 failure mode
- **高速直冲目标后 crash**：检查 `speed_tracking` 是否在近距离强烈减速效果不够，可增大 `lambda_speed` 或减小 `d_ref`。
- **悬停不前**：可能 `speed_tracking` 期望速度过高导致 penalty 过大，需调整 `max_speed` 或 `d_ref`。
- **反复轻触垫圈分**：若 agent 学会只满足 `soft_landing` 的部分条件而忽略定位或速度，需收紧阈值。
- **progress 奖励消失后行为无目标**：在极近目标后 `progress` 渐弱，应靠 `soft_landing` 完成最后的姿态调整。若 `soft_landing` 权重不足，可适度增加。
