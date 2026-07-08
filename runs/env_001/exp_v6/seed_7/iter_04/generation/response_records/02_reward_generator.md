# Response Record

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # next_obs 解包
    x_pos = next_obs[0]
    y_pos = next_obs[1]
    # x_vel = next_obs[2]   # 未直接使用，但保留以备扩展
    # y_vel = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # 1. 距离奖励：基于到目标平台中心的欧氏距离，密集引导接近目标
    distance = (x_pos ** 2 + y_pos ** 2) ** 0.5
    distance_reward = -distance

    # 2. 姿态惩罚：避免倾斜过大，鼓励竖直姿态
    angle_penalty = -abs(angle)

    # 3. 角速度惩罚：抑制快速旋转，提高稳定性
    angular_vel_penalty = -abs(ang_vel)

    # 4. 能量惩罚：减少引擎使用，满足节省燃料要求
    energy_penalty = 0.0
    if action == 2:               # 主引擎
        energy_penalty = -0.1
    elif action in (1, 3):        # 转向引擎
        energy_penalty = -0.05
    # action 0 不额外惩罚

    # 5. 时间惩罚：鼓励尽快完成任务
    time_penalty = -0.01

    # 总奖励为各项直接相加
    total_reward = (
        distance_reward
        + angle_penalty
        + angular_vel_penalty
        + energy_penalty
        + time_penalty
    )

    components = {
        "distance_reward": distance_reward,
        "angle_penalty": angle_penalty,
        "angular_vel_penalty": angular_vel_penalty,
        "energy_penalty": energy_penalty,
        "time_penalty": time_penalty,
        "total_reward": total_reward,
    }

    return float(total_reward), components
```

---

### 设计说明

**主信号选择：distance_reward（绝对值距离惩罚）**  
前序迭代使用的 `progress_delta + landing_bonus + stability_penalty` 已失败，因此本次回避所有增量进步类骨架（包括 `progress_delta_reward` 和 `potential_based_shaping`），改用**静态距离惩罚**作为核心过程引导。每步给予 `-distance`，直接鼓励飞行器始终保持在目标平台附近，避免在目标上方悬停而不下降，同时也对远离目标的行为施加持续性负反馈。

**其他组件角色**  
- **angle_penalty & angular_vel_penalty**：利用 `body_angle` 和 `angular_velocity` 实现姿态稳定，防止翻滚或剧烈旋转，提升最终降落的平稳性。  
- **energy_penalty**：根据动作类型对使用主引擎和转向引擎分别施加小惩罚，响应“尽可能减少引擎推力”的任务要求。  
- **time_penalty**：每步固定负值，促使智能体以最短时间完成到达与稳定，同时抑制无意义的原地逗留。

**未使用的骨架及理由**  
- `terminal_success_reward` / `terminal_failure_penalty`：环境未提供任何显式的 `success` 或 `failure` 标志，信息接口不足以可靠实现。  
- `distance_reward` 的绝对值形式天然隐含了对越界（水平坐标过大）的惩罚，因此无需单独设计 `failure_penalty` 或 `gated_reward`。  
- `landing_bonus` 已在之前试验中被规避，且缺乏可靠的成功检测手段，故本次不予采用。  
- `potential_based_shaping` 虽为推荐候选，但其效果与 `progress_delta` 本质上同类，基于增量变化，为避免重复失败模式而推迟。

**预期需要观察的失败模式**  
- 智能体可能满足于在目标正上方小距离悬停，而迟迟不完成最终着陆（y≈0 但 x≈0 处 hover）。此时 `energy_penalty` 和 `time_penalty` 会持续施压，需观察其能否推动最终接触。  
- 如果着陆后 `body_not_awake_or_settled` 触发较慢，`time_penalty` 会持续累积，可能导致智能体在着陆瞬间仍受负奖励，但整体影响应有限。  
- 距离惩罚可能使智能体在学习早期过分保守，不敢探索更远区域；可通过调整相对权重缓解，但当前最小设计保留单一尺度以观察效果。  

**留待后续迭代的目标**  
- 对最终稳定着陆的显式肯定（如利用双腿接触+低速+小角度的组合推断成功），待观察到 `distance_reward` 主信号能产生可靠趋近行为后，再考虑通过辅助奖励注入，以避免当前强行添加不可靠的软成功奖励。
