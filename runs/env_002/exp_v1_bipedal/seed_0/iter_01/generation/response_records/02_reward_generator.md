# Response Record

# reward_v1.py

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # ========== 主学习信号：前进速度奖励 ==========
    # 使用水平速度(obs[2])作为前进驱动力
    # 鼓励agent向前移动，这是任务的核心目标
    forward_velocity = next_obs[2]  # 使用next_obs避免延迟
    fwd_scale = 2.0
    progress_reward = fwd_scale * forward_velocity
    
    # ========== 存活奖励：鼓励持续行走 ==========
    # 通过主体角度(obs[0])和角速度(obs[1])判断是否还站立
    # 当主体角度过大或角速度过高时，说明即将摔倒，减少存活奖励
    hull_angle = abs(next_obs[0])  # 主体偏离竖直方向的角度
    hull_angular_vel = abs(next_obs[1])  # 主体角速度绝对值
    
    # 存活条件：角度小于0.5弧度(~28度)且角速度小于2.0
    alive_condition = (hull_angle < 0.5) and (hull_angular_vel < 2.0)
    alive_bonus = 0.5 if alive_condition else 0.0
    
    # ========== 稳定性约束：轻量惩罚 ==========
    # 惩罚过大的主体角度和角速度，防止摔倒
    # 使用连续函数，避免二值惩罚导致的梯度消失
    angle_penalty_scale = 0.5
    angular_vel_penalty_scale = 0.3
    
    angle_penalty = angle_penalty_scale * (hull_angle ** 2)
    angular_vel_penalty = angular_vel_penalty_scale * (hull_angular_vel ** 2)
    stability_penalty = -(angle_penalty + angular_vel_penalty)
    
    # ========== 总奖励 ==========
    total_reward = progress_reward + alive_bonus + stability_penalty
    
    # ========== 组件记录 ==========
    components = {
        "progress_reward": progress_reward,
        "alive_bonus": alive_bonus,
        "stability_penalty": stability_penalty
    }
    
    return float(total_reward), components
```

# reward_v1 设计说明

## 使用的奖励组件

1. **progress_reward**（主学习信号）：基于水平速度(obs[2])的稠密前进奖励。这是任务的核心驱动力，告诉agent"向前走就能得分"。使用next_obs中的速度值避免一步延迟，权重2.0使其成为奖励中最大的正向贡献者。

2. **alive_bonus**（存活激励）：当主体角度和角速度在安全范围内时给予小额存活奖励(0.5)。这鼓励agent保持站立状态，避免过早摔倒终止回合。存活条件使用连续阈值判断，比纯二值更平滑。

3. **stability_penalty**（稳定约束）：对主体角度和角速度的平方施加轻量惩罚。这作为"方向盘"引导agent保持直立姿态，防止因过度追求前进速度而摔倒。权重较小(0.5和0.3)，不会压制探索。

## 为什么没有使用 terminal_success_reward / terminal_failure_penalty

- **explicit_success_flag_available = false**：环境info始终为空，没有显式success标志。虽然存在"到达地形终点"的终止条件，但无法在step中可靠检测。
- **explicit_failure_flag_available = false**：同样没有显式failure标志。"身体摔倒"的终止条件只能通过终止事件间接知道，但终止时无法获取next_obs来设计惩罚。
- 这两个组件留到后续迭代，当环境接口提供更丰富的终止信息时再加入。

## 留到后续迭代的组件

- **energy_penalty**：v1阶段先让agent学会稳定行走，后续再优化能耗效率。
- **action_smoothness_penalty**：如果观察到动作抖动，后续迭代再加入。
- **time_penalty**：v1阶段不鼓励agent快速结束回合，后续若发现agent拖沓再考虑。
- **terminal_failure_penalty**：需要环境提供显式failure信号或更丰富的终止信息。
- **gated_reward**：复杂门控机制，v1阶段保持简单。

## 训练后应观察的 failure mode

1. **fast_then_fail**：agent可能快速前进但频繁摔倒。如果观察到这种情况，应增大stability_penalty权重或加入更严格的角度约束。
2. **hover_or_stand_still**：agent可能原地不动获取存活奖励。如果观察到水平速度长期接近0，应减小alive_bonus权重或增加前进速度的最小阈值要求。
3. **over_conservative_policy**：如果stability_penalty过强，agent可能不敢移动。应观察前进速度是否持续偏低，适当降低稳定性惩罚权重。
4. **contact_hacking**：如果agent利用腿部接触信号获取额外奖励（当前未使用contact信号，但后续迭代可能加入），需要警惕。
