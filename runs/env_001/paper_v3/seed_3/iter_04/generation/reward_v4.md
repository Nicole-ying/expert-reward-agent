## 1. evidence
当前score=-25.51，episode_length=971.85，terminated仅1/20、truncated达19/20，说明策略存活满步数但未完成任务；approach_and_soft_landing的episode_sum_mean仅3.98（100% active但几乎无净进展），contact_reward均值31.35但active_rate仅1.1%（几乎不触发），upright_stabilization仅-0.69（姿态保持良好）；上一轮将approach从持久状态值改为势能差分后得分显著恶化，证明delta形式无法为悬停策略提供脱离梯度。

## 2. behavior_diagnosis
策略已陷入悬停局部最优：飞行器在远离目标处保持姿态稳定、速度近零，存活满~972步后被截断，既不坠毁也不接近目标着陆；势能差分在静止时输出零信号，无法产生驱动飞向目标的梯度。

## 3. signal_completeness
必要职责基本完备——趋近引导、软着陆减速、姿态稳定、双足接触奖励均已存在。但approach_target职责的数学形态（势能差分）在策略静止时信号塌缩为零，使总体奖励缺乏从悬停状态脱离的梯度，导致该职责实际上不可达。

## 4. selected_level
Level 2：证据表明势能差分（state_to_improvement）在策略静止时信号为零、无法提供梯度驱动接近行为，需结构变换为持久有界邻近奖励（state_value）；非单纯尺度问题——加大差分系数乘以零仍为零。

## 5. selected_intervention
唯一修改目标组件`approach_and_soft_landing`中的趋近信号：从势能差分`dist_curr - dist_next`变换为持久有界邻近值`3.0/(1.0+dist_next)`；速度惩罚门控和系数保持完全不变；其他两个组件（upright_stabilization、contact_reward）一字不改。

## 6. falsifiable_hypothesis
持久邻近奖励在任意状态（包括静止悬停）下均提供非零梯度——距离越近奖励越高，策略随机探索时偶然靠近目标会立即获得更高的单步奖励，从而打破悬停均衡、重建向目标移动的信用分配路径；接近目标后速度门控惩罚和接触奖励会接管软着陆与完成行为。

## 7. expected_next_round
approach_and_soft_landing的episode_sum_mean应显著上升（持久奖励替代零差分），active_rate保持100%；episode_length可能缩短（策略开始主动接近而非悬停等待截断）；contact_reward的active_rate应从1.1%上升（到达目标区域后接触事件增多）；score应从-25.51改善，truncated比例下降。

## 8. main_risk
持久状态值存在"近目标悬停刷分"风险——若接触奖励3.0不足以覆盖着陆动作风险，策略可能在dist≈0处悬停收集approach_reward≈3.0/步而不完成双足着陆；下一轮需观察contact_reward的active_rate和terminated比例以排除此漏洞。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    """
    Goal: Guide a 2D lander to softly touch down on the target pad with both feet,
    staying upright and near zero velocity.
    
    Key change: approach reward is persistent bounded proximity 3/(1+dist)
    instead of potential-based delta, providing gradient even when stationary.
    """
    # Unpack next observations
    x_next, y_next = next_obs[0], next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # --- Component 1: Approach and soft landing ---
    # Distance to target
    dist_next = (x_next**2 + y_next**2) ** 0.5

    # Persistent bounded proximity: non-zero gradient even when hovering
    approach_reward = 3.0 / (1.0 + dist_next)

    # Velocity penalty gated by proximity: heavier when close
    gate_vel = 1.0 / (1.0 + dist_next)
    vel_penalty = (vx**2 + vy**2) * gate_vel

    w_vel = 0.1
    comp_approach_landing = approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    comp_stabilization = -w_angle * angle_penalty - w_angvel * angvel_penalty

    # --- Component 3: Successful contact reward ---
    w_contact = 3.0
    comp_contact = w_contact * (left_contact * right_contact)

    total_reward = comp_approach_landing + comp_stabilization + comp_contact
    components = {
        "approach_and_soft_landing": comp_approach_landing,
        "upright_stabilization": comp_stabilization,
        "contact_reward": comp_contact
    }
    return float(total_reward), components
```