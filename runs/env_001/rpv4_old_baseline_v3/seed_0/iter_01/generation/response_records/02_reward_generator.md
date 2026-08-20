# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Reward for 2D lander: reach pad, soft landing, stable orientation, minimal fuel.
    v1: dense progress via improvement_delta, orientation penalty,
        soft contact proxy, and near‑pad velocity damping.
    """
    # Extract relevant signals
    x_prev, y_prev = obs[0], obs[1]
    x_curr, y_curr = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angvel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. Progress reward (improvement_delta on Euclidean distance)
    dist_prev = (x_prev ** 2 + y_prev ** 2) ** 0.5
    dist_curr = (x_curr ** 2 + y_curr ** 2) ** 0.5
    progress = dist_prev - dist_curr                     # positive when approaching
    w_progress = 1.0
    progress_reward = w_progress * progress

    # 2. Orientation penalty (quadratic_penalty)
    w_angle = 0.2
    w_angvel = 0.05
    orientation_penalty = -w_angle * (angle ** 2) - w_angvel * (angvel ** 2)

    # 3. Soft contact reward (joint_condition_proxy, continuous)
    contact_flag = left_contact * right_contact          # both feet on ground
    sigma_angle = 0.1
    sigma_vel = 0.2
    # exp(-angle**2 / sigma_angle**2) * exp(-(vx**2+vy**2) / sigma_vel**2)
    angle_factor = 2.718281828 ** (- (angle ** 2) / (sigma_angle ** 2))
    vel_factor = 2.718281828 ** (- (vx ** 2 + vy ** 2) / (sigma_vel ** 2))
    soft_contact = contact_flag * angle_factor * vel_factor

    w_contact = 2.0
    contact_reward = w_contact * soft_contact

    # 4. Near-pad velocity damping (hinge / gated penalty)
    dist_thresh = 1.0
    gate = max(0.0, 1.0 - dist_curr / dist_thresh)       # 0 at far, 1 at origin
    w_vel_damp = 0.5
    vel_damping = -w_vel_damp * gate * (abs(vx) + abs(vy))

    total_reward = progress_reward + orientation_penalty + contact_reward + vel_damping

    components = {
        "progress_reward": progress_reward,
        "orientation_penalty": orientation_penalty,
        "contact_reward": contact_reward,
        "vel_damping": vel_damping
    }
    return float(total_reward), components
```

# reward_v1 设计说明

## selected task_family / dynamics_subtype
- task_family: `navigation_goal_reaching`
- dynamics_subtype: `goal_approach_and_soft_contact`（2D 着陆器，离散推力发动机，两点接触支撑）

## selected reward roles
1. **goal_proximity** – 主学习信号，驱动接近目标着陆垫（mandatory）
2. **safe_orientation** – 稳定/安全约束，抑制大幅度翻滚和角速度（mandatory）
3. **soft_landing_success（重新解释为持续接触奖励）** – 通过双腿接触 + 小角度 + 低速度的连续乘积，奖励正确着陆状态（将原本的终端事件转化为稠密 proxy）
4. **velocity_damping_near_pad** – 靠近垫子时鼓励减速，防止硬着陆（conditional，通过距离门控）

## role_to_signal_mapping

| role | signals used | formula operator |
|------|--------------|------------------|
| goal_proximity | obs[0], obs[1], next_obs[0], next_obs[1] | improvement_delta（每步距离减小奖励） |
| safe_orientation | next_obs[4], next_obs[5] | quadratic_penalty |
| soft_contact (替代 soft_landing_success) | next_obs[6], next_obs[7], next_obs[4], next_obs[2], next_obs[3] | joint_condition_proxy（连续因子乘积） |
| velocity_damping_near_pad | next_obs[2], next_obs[3], next_obs[0], next_obs[1] | hinge / gated penalty（距离门控的线性速度惩罚） |

## excluded roles 及原因
- **terminal_success_reward**：info 无 done 标志，无法可靠判断终止时刻并给予 sparse event reward。已替换为持续 soft contact reward。
- **terminal_failure_penalty**：同样缺少 done 与 termination_reason，且 v1 应避免硬覆盖回报；失败模式通过进度惩罚、速度惩罚等稠密信号间接抑制。
- **action_efficiency**：v1 默认不包含；离散动作的效率项可能导致 agent 不敢使用姿态发动机，留到后续迭代。
- **contact_balance**：只有二元接触标志，无连续力矩/压力信号，无法精确奖励平衡。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty
环境卡片明确 `explicit_success_flag_available=false` 且 info 始终为空，`compute_reward` 没有 done 参数（无法知道 episode 是否终止）。因此任何 terminal event 组件都不可行。对应的职责（软着陆成功）通过持续 joint_condition_proxy 实现——只要双腿接触且姿态与速度良好，就给予奖励。

## 哪些职责留到后续迭代
- 动作效率（action_efficiency）——待 agent 稳定着陆后，引入微小动作惩罚以优化燃料使用。
- 更多动态门控（如根据剩余距离自动调整 orientation 惩罚）——当前固定权重已足够。
- 课程式 reward scaling（training_progress 驱动）——v1 无参考轨迹，暂不使用。

## 训练后应该观察哪些 failure modes
- **盘旋不下降**：progress_reward 不足以克服高度维持，长期 y 较大。检查距离项权重。
- **高速硬着陆**：done 时速度仍很高，可能 crash。需观察 vel_damping 的 gate 范围及权重是否合适。
- **姿态翻滚**：angle 或 angvel 持续偏大，即使接触后也发生姿态崩坏。可能需要提高 orientation 惩罚权重或降低 sigma_angle。
- **漂移出界**：水平位置超出视口，说明 progress 未充分抑制水平速度，可增大 vel_damping 或加入 x_position 的显式惩罚。
- **“单腿着陆”**：只有一侧接触仍试图借重力停住，可能不稳定。contact_reward 要求双接触可引导对称着陆，若依然出现需检查接触阈值。
