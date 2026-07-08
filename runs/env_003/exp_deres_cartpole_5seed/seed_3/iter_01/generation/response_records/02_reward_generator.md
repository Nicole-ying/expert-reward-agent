# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Env_003 生存平衡任务 reward_v1。
    使用连续负惩罚引导摆杆保持竖直并使底座接近轨道中心。
    """
    # 提取下一步观察
    base_pos = next_obs[0]       # 底座水平位置
    pole_angle = next_obs[2]     # 杆偏角 (rad)
    pole_angular_vel = next_obs[3]  # 杆角速度

    # 惩罚系数
    w_angle = 5.0
    w_angular_vel = 0.5
    w_pos = 0.1

    # 核心学习信号：偏离直立和中心的二次惩罚
    progress_reward = -(
        w_angle * (pole_angle ** 2) +
        w_angular_vel * (pole_angular_vel ** 2) +
        w_pos * (base_pos ** 2)
    )

    total_reward = progress_reward
    components = {"progress_reward": progress_reward}

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及角色

- **progress_reward**（唯一组件）：角色为**主学习信号**。它使用连续二次惩罚形式，对杆偏角、杆角速度和底座位置分别施加惩罚，引导 agent 尽快将杆恢复竖直（pole\_angle≈0）、抑制杆的振荡（angular\_vel≈0）并使底座保持在轨道中心（base\_pos≈0）。该信号每步提供稠密梯度，帮助 agent 学习如何用力维持平衡。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- 环境卡片明确声明 **explicit_success_flag_available=false**、**explicit_failure_flag_available=false**，且 info 为空字典，没有任何终止原因字段。  
  因此无法在奖励函数中区分成功/失败终止，强行构造此类奖励将导致不可靠的猜测或伪造信号，故不纳入 v1。

## 为什么没有使用其他骨架

- **distance_reward / progress_delta_reward**：任务无明确目标位置，不适合。
- **stability_penalty** 作为独立组件：其功能已整合到 progress\_reward 中（角度与角速度惩罚），无需额外拆分，避免产生重复信号。
- **energy_penalty / time_penalty**：属于效率约束，v1 阶段应先学会稳定平衡再考虑能耗优化，避免导致 agent 不敢做动作。
- **gated_reward / soft_landing_proxy**：本次任务无明显分阶段或复合完成条件，暂不引入复杂门控。

## 后续迭代预留

- 若 agent 稳定后仍频繁因底座越界失败，可考虑对 `base_pos` 使用更陡的惩罚（如分段线性）或增加底座速度约束（小型 stability\_penalty 拆分），但在 v1 中先用二次惩罚铺底。
- 若出现“接近平衡但持续在边界振荡”的 exploit 行为，后续可引入基于底座接近边界的轻量**边界风险惩罚**。
- 若训练后期想进一步优化动作效率，可添加极小的 **energy_penalty**（基于 action 能耗），但 v1 阶段不会添加。

## 训练后需观察的 failure mode

- **agent 学会让杆倒下更快触发 reset 以逃避惩罚**：虽然负奖励累计会更大，但如果早期随机探索中 agent 发现快速失败能获得较小的累计负奖励（因为步数少），可能形成“主动失败”策略。这是生存任务常见问题。后续可通过增加部分对存活本身的微弱正向奖励或使用 discounted return 来缓解。
- **底座大幅来回移动以补偿杆偏角但不越界**：可能产生高频振荡但不跌倒，奖励仍接近零；需要检查是否在真实环境中这种行为实际无法维持稳定，或通过增加角速度惩罚权重抑制。
- **杆偏角始终维持在临界值附近（如 ±0.2 rad）但仍存活**：此时角度惩罚约为 0.2²×5=0.2，存活 500 步总惩罚约 100，若 agent 发现这是局部最优可能不学习更竖直的控制，后续可考虑使用更敏感的非线性惩罚（如指数形）。
