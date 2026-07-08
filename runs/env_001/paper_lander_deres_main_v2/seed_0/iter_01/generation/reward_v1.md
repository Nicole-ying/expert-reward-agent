# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前观测与下一观测
    x, y = obs[0], obs[1]
    next_x, next_y = next_obs[0], next_obs[1]
    vx, vy = next_obs[2], next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # ---------- 主学习信号：progress delta ----------
    def dist(px, py):
        return (px**2 + py**2)**0.5

    prev_dist = dist(x, y)
    next_dist = dist(next_x, next_y)
    progress_delta = prev_dist - next_dist    # 正值 = 靠近目标

    # ---------- 稳定/安全约束：stability penalty ----------
    speed_penalty = abs(vx) + abs(vy)
    angle_penalty = abs(angle)
    angvel_penalty = abs(angular_vel)
    stability_penalty = -0.1 * speed_penalty - 0.1 * angle_penalty - 0.05 * angvel_penalty

    # ---------- 任务完成近似信号：soft landing proxy ----------
    both_contact = left_contact + right_contact   # 两个都接触时 = 2.0
    # 条件：几乎在目标正上方、低速、姿态正立、双脚触地
    near_target = abs(next_x) < 0.2 and abs(next_y) < 0.2
    low_speed = (abs(vx) + abs(vy)) < 0.15
    upright = abs(angle) < 0.2
    if both_contact == 2.0 and near_target and low_speed and upright:
        soft_landing_proxy = 0.5
    else:
        soft_landing_proxy = 0.0

    # ---------- 合成 ----------
    total_reward = progress_delta + stability_penalty + soft_landing_proxy

    components = {
        'progress_delta': progress_delta,
        'stability_penalty': stability_penalty,
        'soft_landing_proxy': soft_landing_proxy
    }
    return float(total_reward), components
```

# reward_v1 设计说明

- **主学习信号   progress_delta**：密集引导，每一步根据与目标水平/垂直距离的变化给出正奖励（当靠近时），鼓励 agent 持续向目标区域靠近。它与任务核心目标“到达目标着陆垫”直接相关。
- **稳定约束   stability_penalty**：对高速、大倾斜和高角速度施加轻惩罚，作用是迫使 agent 在接近目标时减速、保持姿态竖直，避免因不稳定姿态导致坠毁或错过着陆。权重较小，不会压制进步信号。
- **任务完成近似信号   soft_landing_proxy**：在没有显式成功标志的条件下，用多条件组合（双脚都接触 & 位置接近目标 & 低速 & 竖直）提供一个额外正奖励。它帮助 agent 理解“最终稳定着陆”是比单纯靠近更有价值的状态。该信号仅在严格条件满足时生效，降低被 exploit 的风险。

**未使用 terminal_success_reward / terminal_failure_penalty 的原因**：  
环境卡片明确声明 `explicit_success_flag_available=false` 且 `info` 为空，不存在可供使用的显式成功/失败标志。依赖终止状态不可靠，因此 v1 通过连续信号和软代理来定义成功行为，而不是奖励终点事件。

**留待后续迭代的组件**：  
- 效率/能耗惩罚（energy_penalty）—— v1 重心是先学会安全着陆，过早惩罚动作可能让 agent 不敢点火。  
- 更复杂的门控或课程奖励（gated_reward）—— 当前阶段不需要多阶段结构。  
- 时间惩罚（time_penalty）—— 等基础任务稳定后再引入。

**应观察的 failure mode**：  
1. **high_reward_without_success**：agent 在目标附近悬停或来回震荡，进度奖励高但从不真正触地（或只碰触一侧）。若出现此情况，可收紧 `soft_landing_proxy` 的条件（如更严格的速度/位置阈值），或增加对未触地的惩罚。  
2. **fast crash near goal**：进度奖励可能驱使 agent 快速冲向地面而不减速。`stability_penalty` 应能缓解，但如果权重太低，需适当增大，或引入垂直速度的额外压制项。  
3. **contact reward hacking**：agent 学习到反复轻触垫面来重复获取 `soft_landing_proxy`。当前条件要求双脚触地、低速、竖直且位于中心，重复轻微起落会破坏低速/竖直条件，风险可控。若依然被 exploit，可进一步收紧条件或使用一次性完成信号（需额外状态记忆，不推荐 v1）。