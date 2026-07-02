# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 1. 主学习信号：进度差分奖励（接近目标垫中心）
    dist_obs = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress_delta = dist_obs - dist_next   # 正值表示接近目标

    # 2. 稳定性约束：惩罚高速、大倾角和高角速度
    speed = (next_obs[2] ** 2 + next_obs[3] ** 2) ** 0.5
    angle_abs = abs(next_obs[4])
    angular_speed = abs(next_obs[5])

    w_speed = 0.05
    w_angle = 0.05
    w_angvel = 0.05
    stability_penalty = -(w_speed * speed + w_angle * angle_abs + w_angvel * angular_speed)

    # 3. 任务完成代理：软着陆近似奖励
    near_target = (dist_next < 0.2)
    low_speed = (speed < 0.1)
    stable_angle = (angle_abs < 0.1)
    both_contact = (next_obs[6] == 1.0 and next_obs[7] == 1.0)
    soft_landing = 0.2 if (near_target and low_speed and stable_angle and both_contact) else 0.0

    # 总奖励
    total_reward = progress_delta + stability_penalty + soft_landing

    components = {
        "progress": progress_delta,
        "stability_penalty": stability_penalty,
        "soft_landing_proxy": soft_landing,
        "total_reward": total_reward
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件及其角色

1. **progress（进度差分奖励）** — 主学习信号  
   - 计算当前步与下一步到目标垫中心（0,0）的距离变化。靠近目标时获得正奖励，远离时惩罚。  
   - 这类差分奖励能提供密集、平滑的引导，帮助飞行器快速学会接近目标区域。

2. **stability_penalty（稳定性惩罚）** — 安全/软着陆约束  
   - 对下一步的水平‑垂直合成速度、机体角度和角速度施加小权重惩罚。  
   - 鼓励在接近目标的过程中降低速度、保持竖直姿态，减少剧烈震荡和撞击风险，为最终软着陆创造条件。

3. **soft_landing_proxy（软着陆代理奖励）** — 任务完成近似信号  
   - 当同时满足四个条件时给予小额固定奖励：十分靠近目标、极低速度、近乎竖直、双腿接触。  
   - 作为成功降落的弱信号，帮助强化最终稳定停留行为，但不虚构 success 标志。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

环境输出中既没有显式的 `success` 标志，也没有 `failure` 标志（`info` 为空），无法可靠地判断终止原因。  
直接使用这两个奖励项会导致模型伪造不可用的信息或误判终止状态。因此根据设计原则，v1 不引入任何依赖显式成功/失败标志的组件。

## 留到后续迭代的组件

- **energy_penalty（能耗惩罚）**：当前 focus 是学习“到达并稳定”，过早加入动作代价可能导致智能体不敢移动或不敢点火。在接近行为基本稳定后，再引入以优化燃料效率。
- **time_penalty（时间惩罚）**：若智能体能接近目标但徘徊过久，后续可加小权重步数惩罚以奖励更快的着陆。
- **gated_reward（门控奖励）** 与 **dynamic curriculum**：在安全性或阶段划分更明确时，可以逐步引入更复杂的多阶段奖励结构。

## 训练后应观察的 failure mode

- **目标附近震荡不接触**：进度奖励可能使飞行器在垫片上方来回摆动，仅靠稳定性惩罚不足以使其真正接触。此时可通过增加 `soft_landing_proxy` 权重或逐步收紧阈值引导着陆。
- **过快撞击或姿态失稳**：若稳定性惩罚过弱，飞行器可能高速撞击目标或倾覆。需要观察速度、角度和角速度的 episodes 统计，必要时增加对应系数。
- **过早退出接近区域**：进度奖励可能使智能体因速度过快而冲出目标区域，导致距离先减后增。如频繁出现，可考虑限速或对大幅反向动作进行额外惩罚。
- **双腿未完全接触即满足 soft_landing 条件**：当前条件要求双腿接触，相对严格。若发现智能体倾向于单腿着陆或悬停不落，应检查接触条件并调整 soft_landing 阈值。