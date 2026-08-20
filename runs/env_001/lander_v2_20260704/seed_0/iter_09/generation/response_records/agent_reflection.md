# Response Record

# 诊断与干预

## 1. 证据与行为诊断

`evidence`：Score=215.30，terminated=13/20，truncated=7/20，length=577.65。contact_reward episode_sum_mean=193.9（signed_share=78.7%），proximity_improvement=39.3（16.0%），velocity_penalty=-10.25（4.2%），angle_penalty=-1.22（0.5%）。contact_reward active_rate=57.6%，说明约332步有接触，且episode_sum_mean≈0.6×332暗示双腿触地后长期停留。

`behavior_diagnosis`：Agent已能在多数episode中着陆（13/20 terminated），但7/20被截断且平均长度577步偏长。contact_reward以78.7%的份额主导总奖励，其「每步持续领取」形态使Agent降落后只需维持接触即可大量累积奖励，缺乏快速完成任务的动力；35%截断率说明部分episode迟迟无法触发终止条件。

## 2. 信号完备性

`signal_completeness`：proximity_improvement提供过程引导（active_rate=98.9%），velocity_penalty和angle_penalty提供稳定约束，contact信号提供着陆目标——职责基本完备。但contact_reward的数学形态是持续状态奖励（per-step），与「尽快着陆后终止」的外部任务目标错位：占据好状态即可持续获奖，形成proxy exploitation。

## 3. 干预层级选择

`selected_level`：**Level 2**。触发条件：证据模式「占据好状态即可持续获奖」直接否定contact_reward的当前数学形态——这不是尺度问题，是结构问题。contact_reward的per-step持续奖励与任务希望「着陆后即终止」的语义冲突，仅调系数无法消除「停留刷分」的激励。

## 4. 干预目标

`selected_intervention`：将contact_reward从**persistent state reward → transition-based event reward**（persistent_to_transition_event）。使用`max(0, next_contact - current_contact)`检测接触状态从0→1的转移，仅奖励「获得接触」的瞬间而非「维持接触」。系数从0.3（per-step）调整为40.0（per-transition-event），与新值域匹配：每条腿首次触地各奖励40，双足共80，与proximity_improvement（~39/episode）尺度可比。

## 5. 可证伪假设

`falsifiable_hypothesis`：移除「维持接触即获奖」后，Agent不再有动机在着陆后拖延，应更快触发终止条件 → episode_length下降，truncated比例下降，contact_reward episode_sum_mean大幅降低（不再累积），proximity_improvement成为主导信号且外部分数维持或改善。

`expected_next_round`：episode_length从577降至更低（理想<400），truncated从7/20减少，contact_reward episode_sum_mean降至~50-80范围，signed_share降至20-30%，proximity_improvement signed_share升至主导地位，score保持或上升。

`main_risk`：过渡奖励可能使Agent学会「碰触即离」（touch-and-go），仅短暂触地而不稳定停留。但proximity_improvement在离开原点时变负、velocity_penalty抑制高速弹跳，应共同约束此行为。若出现此问题，下一轮可考虑将contact transition与angle条件联合。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # 当前步与下一步的相对目标距离
    dist_current = (obs[0] ** 2 + obs[1] ** 2) ** 0.5
    dist_next = (next_obs[0] ** 2 + next_obs[1] ** 2) ** 0.5

    # 非线性接近度：有界且近目标时梯度更强
    prox_current = 1.0 / (1.0 + dist_current)
    prox_next = 1.0 / (1.0 + dist_next)

    # 状态改善量：奖励靠近，惩罚远离，停留则趋零
    proximity_improvement = 80.0 * (prox_next - prox_current)

    # 稳定约束：抑制高速
    x_vel = next_obs[2]
    y_vel = next_obs[3]
    velocity_penalty = -0.1 * (abs(x_vel) + abs(y_vel))

    # 姿态约束：抑制大幅倾斜
    angle_penalty = -0.05 * abs(next_obs[4])

    # 着陆信号：仅在接触从0→1的转移步给予奖励，而非持续维持
    contact_gain_left = max(0.0, next_obs[6] - obs[6])
    contact_gain_right = max(0.0, next_obs[7] - obs[7])
    contact_reward = 40.0 * (contact_gain_left + contact_gain_right)

    total_reward = proximity_improvement + velocity_penalty + angle_penalty + contact_reward

    components = {
        'proximity_improvement': proximity_improvement,
        'velocity_penalty': velocity_penalty,
        'angle_penalty': angle_penalty,
        'contact_reward': contact_reward
    }

    return float(total_reward), components
```
