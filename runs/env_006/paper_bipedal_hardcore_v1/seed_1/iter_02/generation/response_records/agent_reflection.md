# Response Record

# 设计理由
当前奖励仅用水平速度推进、躯干倾斜角 hinge 惩罚和动作效率惩罚。从训练反馈看，progress_reward 累积值高（145），说明 agent 学会了快速前进；但真实环境得分仅 -18，且所有 episode 都以 terminated 结束（411步均值，无截断），说明 agent 在快速移动中频繁摔倒，摔倒前的角速度突变未被感知。观测审计发现 **hull_angular_velocity（obs[1]）** 未被使用，该信号能提前预示躯干剧烈旋转，是摔倒的早期预警。  
因此新增 `angular_velocity_penalty` 组件，对躯干角速度施加二次惩罚，抑制可能导致摔倒的急速旋转。系数设为 -0.1，预期 per-step 惩罚 ≈ -0.025（假设中等角速度），远低于主信号 per-step（≈0.35）的 0.3 倍，符合校准要求。其它组件保持不变，仅增加这一个结构变换，解决信号缺口。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 主进展：水平速度正向奖励（保持原样）
    horizontal_speed = obs[2]
    progress_reward = 1.0 * horizontal_speed

    # 稳定约束：躯干倾斜角 hinge 惩罚（保持原样）
    hull_angle = obs[0]
    angle_threshold = 0.3
    hinge = max(0.0, abs(hull_angle) - angle_threshold)
    stability_penalty = -0.5 * hinge

    # 新增：躯干角速度二次惩罚（早期摔倒预警）
    hull_angular_velocity = obs[1]
    angular_velocity_penalty = -0.1 * (hull_angular_velocity ** 2)

    # 效率代价：动作二次惩罚（保持原样）
    action_efficiency_penalty = -0.01 * sum(a * a for a in action)

    total_reward = progress_reward + stability_penalty + angular_velocity_penalty + action_efficiency_penalty
    components = {
        'progress_reward': progress_reward,
        'stability_penalty': stability_penalty,
        'angular_velocity_penalty': angular_velocity_penalty,
        'action_efficiency_penalty': action_efficiency_penalty
    }
    return float(total_reward), components
```

# 诊断摘要
- **audit**: 信号缺口：hull_angular_velocity 未被使用，缺少摔倒早期预警；添加轻量二次惩罚以利用该观测。
- **behavior**: agent 快速前进但易摔倒，导致真实得分极低（-18），所有 episode 以终止结束。
- **signal**: 缺躯干角速度抑制信号；现有倾斜角 hinge 惩罚触发率低（19.6%），预防不足。
- **level**: Level 2 （结构变换：添加新组件）
- **hypothesis**: 惩罚快速角速度将使策略学习避免剧烈旋转，减少摔倒概率，从而提升真实环境得分。
- **risk**: 若正常步态伴随小幅角速度，轻量惩罚可能轻微拖慢学习进度；系数 0.1 足够小，预计可承受。
