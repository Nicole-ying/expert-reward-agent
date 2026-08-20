# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ---- 主学习信号：distance-based progress（只奖励接近，不惩罚远离） ----
    prev_distance = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    next_distance = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5
    progress = prev_distance - next_distance
    progress_reward = max(progress, 0.0) * 2.0  # 权重 2.0，鼓励有效靠近

    # ---- 稳定/安全约束：抑制高速、大角度和高角速度 ----
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    angular_vel = next_obs[5]

    # 轻量级线性惩罚，防止剧烈翻滚和高速撞击
    stability_penalty_value = (
        0.1 * abs(vx)
        + 0.1 * abs(vy)
        + 0.1 * abs(angle)
        + 0.05 * abs(angular_vel)
    )
    # 组件记录为负贡献
    stability_penalty = -stability_penalty_value

    # ---- 任务完成近似信号：多条件组合的软着陆 bonus ----
    # 条件：双脚接触 + 位置接近垫中心 + 低速 + 姿态平稳
    soft_landing_bonus = 0.0
    if (next_obs[6] == 1.0 and next_obs[7] == 1.0
            and abs(next_obs[0]) < 0.3
            and abs(next_obs[1]) < 0.1
            and abs(vx) < 0.3
            and abs(vy) < 0.3
            and abs(angle) < 0.2):
        soft_landing_bonus = 1.0  # 为完成提供明确的溢价信号

    # ---- 总奖励 ----
    total_reward = progress_reward + soft_landing_bonus + stability_penalty

    components = {
        "progress_reward": progress_reward,
        "stability_penalty": stability_penalty,  # 负值
        "soft_landing_bonus": soft_landing_bonus
    }

    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

- **progress_reward**（主学习信号，必须包含）  
  采用 `max(prev_distance - next_distance, 0)`，只奖励每步向目标垫中心的靠近，不因暂时远离而惩罚智能体。权重设为 2.0，提供稠密的、与任务目标直接相关的梯度。  
  角色：密集引导智能体朝坐标原点（即着陆垫上方）移动。

- **stability_penalty**（稳定约束，允许 0~2 个）  
  以线速度、机身角度和角速度的绝对值为基础，施加轻量级惩罚：`0.1·|vx| + 0.1·|vy| + 0.1·|angle| + 0.05·|ω|`。  
  角色：抑制高速坠落、剧烈翻滚，促使智能体学习流畅、平稳的接近过程，同时避免因过强的速度抑制导致“不敢行动”。权重显著低于主信号，确保不会压制接近行为。

- **soft_landing_bonus**（任务完成近似信号，允许 0~1 个）  
  当同时满足双脚接触、位置靠近垫中心、低速度和姿态平稳等条件时，给予一次性 1.0 bonus。  
  角色：因为环境没有显式成功标志，利用多条件组合构造一个合理的“近似完成”信号，帮助智能体在靠近后完成最后的软着陆，而不是在目标附近震荡。该信号仅用接触作为条件之一，并且与位置、速度和姿态组合，避免了单纯接触带来的 reward hacking。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

依 `environment_card.md` 声明，`explicit_success_flag_available` 和 `explicit_failure_flag_available` 均为 `false`，且 step 返回的 `info` 为空字典。若在此条件下强行使用终端成功/失败奖励，必须凭空捏造不存在的信号字段，这违反 reward 设计的约束。因此 v1 完全放弃了终端事件奖励，转而用稠密的 progress 和 soft landing bonus 来驱动任务完成。

## 留到后续迭代的组件

- **energy_penalty** 与 **time_penalty**：当前 v1 以“学会完成任务”为首要目标，过早引入能耗或时间代价可能会导致智能体不敢使用引擎、无法学习有效策略。待智能体稳定着陆后，再逐步加入燃料/时间优化项。
- **gated_reward / dynamic_curriculum**：复杂门控和课程学习在 v1 中不加，可以后续针对着陆过程的分阶段引导或防撞课程引入。
- **terminal_success_reward**：若未来 wrapper 能提供显式成功标志，可以替换或补充 soft_landing_bonus。

## 训练后应观察的 failure mode

1. **目标附近缓慢盘旋不下落**：若 progress 和 soft_landing_bonus 之间的梯度衔接不好，智能体可能学会在 (0,0) 附近徘徊以获取 progress_reward，但迟迟不进入软着陆条件。此时可考虑降低 soft_landing 阈值或提高 bonus 值，或逐步收紧 progress 的允许范围。
2. **高速俯冲后撞击**：若 stability_penalty 权重过小，智能体可能牺牲姿态稳定换取更快的距离减少，导致最终坠毁。此时应适度提高速度/角度惩罚系数。
3. **单脚触地反复弹跳**：soft_landing_bonus 的双脚接触条件可以防止单脚支撑 exploit，但如果阈值设置过宽，智能体可能通过反复轻触垫面来累积 bonus。观察到这种模式时，应收紧速度或距离条件。
4. **过度保守、移动缓慢**：进度极慢可能是因为 stability 过强，压低速度惩罚或降低权重即可缓解。
5. **对初始扰动敏感**：随机初始力可能导致部分 episode 距离过远，progress 信号稀疏。若早期学习困难，可考虑在后续迭代中加入逐 step 的 negative distance 辅助，但需注意避免与 progress 产生重复梯度。
