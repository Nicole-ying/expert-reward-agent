1. `evidence`：第一轮 score=-214.16，goal_approach 是唯一激活组件（per-step 仅 ~0.0008），soft_landing 与 crash_avoidance 激活率为 0%，说明 agent 在所有评估回合中从未触发速度惩罚和坠毁惩罚，行为完全由极弱的距离奖励驱动，导致快速失控坠毁。
2. `behavior_diagnosis`：agent 尝试减少距中心距离，但因缺乏速度与姿态约束，迅速加速撞地或飞出边界，回合在 182 步即终止，无成功着陆迹象。
3. `signal_completeness`：任务需要可达的速度/姿态约束和着陆引导；现有 goal_approach 仅提供位置接近，soft_landing 的“仅低空激活”门控使其在高空完全失效，着陆信号缺失，必须补充全程速度/姿态惩罚。
4. `selected_level`：Level 2（修改组件数学形态），因为 soft_landing 的触发条件门控导致其死组件，需改为无门控的持续惩罚以提供可学习的速度控制信号。
5. `selected_intervention`：唯一目标组件为 soft_landing，将其从“低空门控二次惩罚”改为“全高度线性惩罚”，同时纳入姿态角度和角速度，使其在全程产生负向激励，引导 agent 学习减速与稳定姿态。
6. `falsifiable_hypothesis`：加入全程速度/姿态惩罚后，agent 将学会使用主引擎抵消下沉、利用姿态引擎抑制翻滚，从而降低惩罚，并与 goal_approach 共同引导向中心垫缓慢降落，最终提高 score 并可能产生腿接触。
7. `expected_next_round`：下一轮 soft_landing 的 active_rate 应升至近 100%，episode_sum_mean 负值明显，goal_approach 仍保持正向但可能因速度控制而步数增加；score 应较当前 -214 有所回升（负值减小或转正），回合长度可能延长，terminated 中早期坠毁减少。
8. `main_risk`：速度惩罚过强可能压制 goal_approach，导致 agent 完全静止或悬停不动、不向目标移动；后续可能需要调整系数或增加着陆成功奖励。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack current observations
    x_pos = obs[0]
    y_pos = obs[1]
    # Unpack next observations
    x_pos_next = next_obs[0]
    y_pos_next = next_obs[1]
    x_vel_next = next_obs[2]
    y_vel_next = next_obs[3]
    body_angle_next = next_obs[4]
    ang_vel_next = next_obs[5]
    left_contact_next = next_obs[6]
    right_contact_next = next_obs[7]

    # 1. goal_approach: potential-based shaping (distance to center)
    dist_current = (x_pos ** 2 + y_pos ** 2) ** 0.5
    dist_next = (x_pos_next ** 2 + y_pos_next ** 2) ** 0.5
    goal_approach = dist_current - dist_next   # positive when approaching

    # 2. soft_landing: full-range velocity, angle, angular velocity penalty
    #    Encourage low speed, upright posture, and slow angular motion.
    vel_penalty = abs(x_vel_next) + abs(y_vel_next)
    angle_penalty = abs(body_angle_next)       # rad, should be near 0
    ang_vel_penalty = abs(ang_vel_next)
    soft_landing = -0.2 * vel_penalty - 0.2 * angle_penalty - 0.2 * ang_vel_penalty

    # 3. crash_avoidance: unchanged (penalise hard crash below pad without both legs)
    BOTH_LEGS = (left_contact_next > 0.5) and (right_contact_next > 0.5)
    CRASH_PENALTY = -10.0
    if y_pos_next < -0.2 and not BOTH_LEGS:
        crash_avoidance = CRASH_PENALTY
    else:
        crash_avoidance = 0.0

    total_reward = goal_approach + soft_landing + crash_avoidance

    components = {
        "goal_approach": goal_approach,
        "soft_landing": soft_landing,
        "crash_avoidance": crash_avoidance,
    }

    return float(total_reward), components
```