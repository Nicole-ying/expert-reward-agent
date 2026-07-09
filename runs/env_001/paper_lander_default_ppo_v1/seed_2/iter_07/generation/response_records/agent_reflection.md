# Response Record

`evidence`：Score 278（超 target 200），20/20 全部正常终止着陆，平均长度 231 步。landing_approach 占 magnitude_share 75.3%，为持续状态型稠密奖励。任务描述明确要求“尽量减少引擎推力消耗”，但当前奖励中无任何引擎使用成本组件，构成信号缺失。上一轮 iter 6 为 iter 5 的完全重复，被拒绝。

`behavior_diagnosis`：Agent 已学会稳定着陆，但在没有燃料效率信号的情况下，可能使用了超出必要的引擎推力。landing_approach 每步持续给予约 0.11 的接近奖励，可能掩盖低效的引擎使用模式。

`signal_completeness`：缺少引擎/燃料效率信号。任务明确要求“尽量减少引擎推力消耗”，但现有四个组件（距离进步、稳定性、着陆接近、时间惩罚）均不涉及引擎使用成本。其他信号基本完备。

`selected_level`：Level 2 — 信号完备性缺口（缺失引擎效率职责），由任务描述直接支持。

`selected_intervention`：新增 `engine_penalty` 组件，对离散动作施加差异化引擎成本：主引擎（action 2）每步 -0.03，姿态引擎（action 1, 3）每步 -0.01，不点火（action 0）为 0。其他四个组件结构与系数完全保持不变。

`falsifiable_hypothesis`：引入引擎成本后，Agent 应减少不必要的引擎激活，同时维持着陆成功率。若 score 大幅下降或 terminated 比例降低，说明惩罚过强抑制了必要的减速操作。

`expected_next_round`：engine_penalty 将显示非零 active_rate（引擎使用时），episode_sum_mean 为适度负值（估计 -2 到 -5 量级）。score 应维持或略有波动，terminated 率应保持 20/20。若 episode length 缩短且终止率不降，说明效率改善。

`main_risk`：主引擎惩罚可能使 Agent 在着陆减速阶段回避必要推力，导致 crash。但 -0.03 系数远小于近距离时 landing_approach 的约 0.1-0.2/步，不会逆转正确决策。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    prev_x, prev_y = obs[0], obs[1]
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    omega = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    prev_dist = (prev_x ** 2 + prev_y ** 2) ** 0.5
    curr_dist = (x ** 2 + y ** 2) ** 0.5
    reward_dist = 2.0 * (prev_dist - curr_dist)

    reward_stability = (
        -0.008 * abs(vx) -
        0.008 * abs(vy) -
        0.008 * abs(angle) -
        0.008 * abs(omega)
    )

    prox = max(0.0, 1.0 - curr_dist / 2.0)
    vel_mag = abs(vx) + abs(vy)
    vel_factor = max(0.0, 1.0 - vel_mag / 2.0)
    ang_factor = max(0.0, 1.0 - abs(angle) / 0.5)

    approach_quality = (prox * vel_factor * ang_factor) ** (1.0 / 3.0)

    both_legs = 1.0 if (left_contact > 0.5 and right_contact > 0.5) else 0.0
    contact_factor = 0.3 + 0.7 * both_legs

    reward_landing = 0.3 * approach_quality * contact_factor

    time_penalty = -0.02

    if action == 0:
        engine_penalty = 0.0
    elif action == 2:
        engine_penalty = -0.03
    else:
        engine_penalty = -0.01

    total_reward = reward_dist + reward_stability + reward_landing + time_penalty + engine_penalty

    components = {
        "distance_progress": reward_dist,
        "stability_penalty": reward_stability,
        "landing_approach": reward_landing,
        "time_penalty": time_penalty,
        "engine_penalty": engine_penalty
    }

    return float(total_reward), components
```
