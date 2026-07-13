根据训练反馈，上一个奖励函数存在三个致命缺陷：
1. **着陆引导完全缺失**：`soft_landing_penalty`的激活门限`gate_landing = max(0, 1 - dist_obs / 1.0)` 在整个评估中保持为0（active_rate=0%），说明距离阈值过小，agent永远无法激活该组件，导致没有任何信号鼓励减速着陆。
2. **姿态惩罚失控**：`upright_penalty`每步都在惩罚角度（active_rate=100%），其二次惩罚形态对任何微小角度都产生代价，累计负奖励压倒了稀疏的正向进度信号（signed_share=60% vs 40%），令agent倾向于不做大范围位移而尽量减少旋转，从而无法朝目标飞行。
3. **进度信号稀疏**：`approach_reward`的差分正奖励只在一小部分步中激活（active_rate=5.9%），说明agent极少缩短与目标的距离，整体行为是原地挣扎或飞离目标，最终大部分episode以坠毁/超出视口终止（terminated=17/20）。

最佳得分（-10.16）来自迭代1的`contact_reward + gated_goal_reward`结构，该结构直接奖励着陆接触和接近目标的状态。而后续所有版本丢弃了这些核心信号，得分急剧恶化到-5510。因此，本轮应以恢复到“接触奖励+目标接近奖励”为核心，同时加入轻量级的引擎罚款（满足节约燃料要求）和更宽容的姿态约束（仅在大角度时惩罚，允许小幅摇摆）。

**修改计划：**
- 添加**双足着陆奖励**：基于`next_obs[6]`和`next_obs[7]`，两腿同时接触时给予每步正向常数奖励，直接引导着陆。
- 添加**目标接近状态奖励**（凸化）：使用`max(0, 1 - dist_next / D)`，让距离越近奖励越大，转为密集梯度信号。
- 保留**差分进度奖励**，维持朝向目标的移动反馈。
- 添加**引擎使用惩罚**：动作不为0时施加小负值，鼓励节约燃料。
- 将**角度惩罚**改为hinge形式：`max(0, abs(angle) - threshold)`，阈值内零惩罚，避免抑制小幅姿态调整。

这些改动有明确的历史证据支持（接触+目标奖励在迭代1有效），并通过对主动率、幅度份额的分析修复了当前组件的失效问题。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 位置和距离
    x_obs, y_obs = obs[0], obs[1]
    x_next, y_next = next_obs[0], next_obs[1]
    dist_obs = (x_obs**2 + y_obs**2)**0.5
    dist_next = (x_next**2 + y_next**2)**0.5

    # 接触状态（基于 next_obs，反映动作后的着陆情况）
    left_contact = next_obs[6]
    right_contact = next_obs[7]
    both_landed = (left_contact > 0.5) and (right_contact > 0.5)

    # 姿态角度
    angle_next = next_obs[4]

    # ========== 组件定义 ==========

    # 1. 双足着陆奖励：每步给予常数奖励，鼓励保持接触直到终止
    w_contact = 1.0
    contact_reward = w_contact * (1.0 if both_landed else 0.0)

    # 2. 目标接近状态奖励（凸化，密集梯度）
    goal_threshold = 2.0
    proximity = max(0.0, 1.0 - dist_next / goal_threshold)
    w_proximity = 0.5
    proximity_reward = w_proximity * proximity

    # 3. 差分进度奖励（保持原有正向差分思想）
    progress = max(0.0, dist_obs - dist_next)
    w_progress = 1.5
    progress_reward = w_progress * progress

    # 4. 引擎使用惩罚（节约燃料）
    w_engine = 0.03
    engine_penalty = -w_engine if action != 0 else 0.0

    # 5. 姿态角度惩罚（hinge，允许小幅摇摆）
    allowed_angle = 0.2  # 约11.5度
    excess_angle = max(0.0, abs(angle_next) - allowed_angle)
    w_angle = 0.1
    upright_penalty = -w_angle * (excess_angle ** 2)

    # 总奖励
    total_reward = contact_reward + proximity_reward + progress_reward + engine_penalty + upright_penalty

    reward_components = {
        'contact_reward': contact_reward,
        'proximity_reward': proximity_reward,
        'progress_reward': progress_reward,
        'engine_penalty': engine_penalty,
        'upright_penalty': upright_penalty
    }

    return float(total_reward), reward_components
```