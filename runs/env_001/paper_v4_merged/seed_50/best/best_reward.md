# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置距离
    current_dist = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_dist = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 进度奖励：距离减小的量（鼓励靠近目标点）
    progress = current_dist - next_dist
    progress_reward = 2.0 * progress

    # 速度惩罚：抑制冲击速度（二次惩罚）
    velocity_penalty = 0.05 * (next_obs[2] ** 2 + next_obs[3] ** 2)

    # 姿态惩罚：抑制大幅倾斜（二次惩罚）
    angle_penalty = 0.1 * (next_obs[4] ** 2)

    # 软着陆近似奖励：同时满足双腿接触、靠近中心、低速、小角度时给予正向信号
    contact = next_obs[6] * next_obs[7]  # 1.0 仅当双腿都接触
    pos_factor = max(0.0, 1.0 - next_dist / 0.5)
    vel_sum = abs(next_obs[2]) + abs(next_obs[3])
    vel_factor = max(0.0, 1.0 - vel_sum / 0.5)
    angle_factor = max(0.0, 1.0 - abs(next_obs[4]) / 0.2)
    soft_landing = 0.5 * contact * pos_factor * vel_factor * angle_factor

    total_reward = progress_reward - velocity_penalty - angle_penalty + soft_landing
    components = {
        "progress_reward": progress_reward,
        "velocity_penalty": velocity_penalty,
        "angle_penalty": angle_penalty,
        "soft_landing_proxy": soft_landing
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **selected task_family / dynamics_subtype**: `navigation_goal_reaching / goal_approach_and_soft_contact`（二维着陆器定点软着陆）。
- **selected reward roles**:
  - 主学习信号：`progress_reward`（基于距离减少的差分奖励）
  - 稳定/安全约束：`velocity_penalty`（速度抑制）、`angle_penalty`（姿态抑制）
  - 任务完成近似：`soft_landing_proxy`（多条件乘积的软成功信号）
- **role_to_signal_mapping**:
  - 进度奖励：使用 `obs[0],obs[1]` 和 `next_obs[0],next_obs[1]` 计算欧氏距离，取其减少量作为改进差分。
  - 速度惩罚：使用 `next_obs[2],next_obs[3]`，二次惩罚。
  - 角度惩罚：使用 `next_obs[4]`，二次惩罚。
  - 软着陆代理：使用 `next_obs[6],next_obs[7]`（双腿接触）、`next_obs[0],next_obs[1]`（距离）、`next_obs[2],next_obs[3]`（速度大小）、`next_obs[4]`（倾角），构造乘积式连续因子。
- **formula operators**:
  - `improvement_delta`（3.2）用于进度奖励，因为靠近目标的距离变化比绝对位置值更直接地反映进展，并避免了在原点静止时获得被动的正信号。
  - `dense_state_signal` 的二次惩罚形式（3.1）用于速度和角度约束，对偏离零的值给出连续梯度，且权重较小以免压制探索。
  - `joint_condition_proxy`（3.8）的连续乘积形式用于软着陆代理：每个条件（接触、位置、速度、倾角）被转化为有界因子后相乘，只有当所有条件同时接近满足时才给出显著奖励，防止任一子因素塌缩到零时误导策略。
- **excluded roles**:
  - `terminal_success_reward` / `terminal_failure_penalty`：环境没有显式成功/失败标志（`info`为空，且无法在 `compute_reward` 中访问 `done` 信号），因此基于终止时一次性奖励的算子不适用。
  - `action_efficiency`：离散动作空间，节能为次要目标，v1 阶段优先学习着陆技能，效率约束留到后续迭代。
  - 存活约束、生存奖励：任务不是生存型，无存活计时器。
- **为什么没有使用 terminal_success_reward / terminal_failure_penalty**：`explicit_success_flag_available=false`，`info` 无任何可用字段，无法在奖励函数内可靠地判断片段是否结束，因此忽略所有基于 `done` 的信号。
- **哪些职责留到后续迭代**：
  - 动作效率 / 能耗（减少发动机使用）——待着陆稳定后按需加入轻量惩罚。
  - 更精细的 `soft_health_gate`（结合距离与速度的预期减速距离）——若 agent 频繁在最后一刻无法减速，可衰减进度奖励。
  - 动态阈值或课程式调整——若早期回合极短，可能需要对阈值或权重进行渐进缩放。
- **训练后应该观察哪些 failure modes**：
  - 高速撞击：进度奖励主导时 agent 可能快速冲向原点，需要监控接触前速度分布。
  - 悬停不前：若速度惩罚过强，agent 可能不愿移动而导致进度停滞；观察距离下降速率。
  - 振荡不触地：双腿未接触但 agent 在原点附近往返；检查软着陆代理信号的触发频率，若几乎不触发可考虑放宽阈值或改用几何平均。
  - 低频触发下的稀疏性问题：软着陆代理只有在接触且位置/速度/角度都在阈值内时才非零；初期可能极少触发，需关注该组件的 `active_rate` 并适时调整门槛或使用连续化更强的因子。