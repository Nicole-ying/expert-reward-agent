`evidence`: 上一轮将approach从状态值改为势能差（potential-based delta），训练后所有episode均被truncated（1000步），terminated=0，contact_reward触发率为0%，agent从未着陆；approach_and_soft_landing均值为-0.187（势能差被速度惩罚略微压负），upright_stabilization均值为-0.361且每步持续扣分，总分-53.78表明agent在徘徊/悬停中耗尽时间。

`behavior_diagnosis`: agent学会了在高空悬停或缓慢漂移来最小化综合惩罚——势能差奖励在恒定距离处趋近于零，而stabilization惩罚每步固定扣除约-0.36，远超approach能提供的正向信号（全程势能差上限仅~4.7）。agent从未下降到足以触发contact的高度，因为任何靠近目标的速度都会被proximity-gated velocity penalty惩罚，且stabilization的持续放血使得探索性下降在回报上不可行。

`signal_completeness`: 四个必要职责（approach、soft_landing_velocity、upright_stabilization、contact）均以某种形态存在，但upright_stabilization是无界的每步惩罚，与有界势能差approach形成结构性尺度错配——stabilization在远离目标时仍全额扣分，压制了早期下降探索。contact信号完备但不可达（active_rate=0%）。

`selected_level`: Level 2 — `global_to_local_gate`。当前stabilization组件是全局恒定惩罚，而证据（episode达到1000步截断、无坠毁、无着陆）表明它在远离目标时阻碍了必要的速度探索。需要将stabilization改为距离门控，使姿态约束只在接近目标时生效。

`selected_intervention`: 唯一目标组件`upright_stabilization`。以best代码为基础（state-based approach），对该组件增加距离门控：`gate_stab = max(0.0, 1.0 - dist/10.0)`，使角度和角速度惩罚在dist≥10时完全归零，在dist→0时线性恢复至全额。不修改approach_and_soft_landing和contact_reward。

`falsifiable_hypothesis`: 距离门控后，agent在远离目标时不再被姿态惩罚放血，可以自由使用主发动机和姿态发动机下降到低空；当dist<10后stabilization逐步激活，提供着陆前的姿态整形信号。state-based approach_reward提供持续的"靠近即得分"引导，配合接触奖励的强正反馈（3.0/step），agent应能发现着陆行为。预期contact_reward的active_rate从0%上升，episode中出现terminated（成功着陆触发机体静止终止）。

`expected_next_round`: upright_stabilization的magnitude_share下降（因远距离时惩罚归零），approach_and_soft_landing的episode_sum_mean转正（agent实际靠近目标），contact_reward的active_rate>0，score显著上升，可能出现terminated>0的episode。

`main_risk`: state-based approach在dist较小时可持续获得奖励，若contact_reward的发现仍然困难（需要双足同时接触），agent可能在低空形成新的悬停平台（在dist≈1-3处徘徊赚取approach奖励但不完成着陆）。此时下一轮需考虑`dense_to_task_event`或增加completion-gated shaping。

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack next_obs (post-action state)
    x = next_obs[0]
    y = next_obs[1]
    vx = next_obs[2]
    vy = next_obs[3]
    angle = next_obs[4]
    ang_vel = next_obs[5]
    left_contact = next_obs[6]
    right_contact = next_obs[7]

    # Distance to target pad at (0,0)
    dist = (x**2 + y**2) ** 0.5

    # --- Component 1: Approach and soft landing (from best, unchanged) ---
    approach_reward = 1.0 / (1.0 + dist)
    gate_vel = 1.0 / (1.0 + dist)
    vel_penalty = (vx**2 + vy**2) * gate_vel
    w_approach = 1.0
    w_vel = 0.1
    comp_approach_landing = w_approach * approach_reward - w_vel * vel_penalty

    # --- Component 2: Upright stabilization (MODIFIED: distance-gated) ---
    angle_penalty = angle**2
    angvel_penalty = ang_vel**2
    w_angle = 0.1
    w_angvel = 0.1
    # Gate: no stabilization penalty beyond 10 units; ramps linearly to full near target
    gate_stab = max(0.0, 1.0 - dist / 10.0)
    comp_stabilization = gate_stab * (-w_angle * angle_penalty - w_angvel * angvel_penalty)

    # --- Component 3: Successful contact reward (unchanged) ---
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