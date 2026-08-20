```json
{
  "evidence": "当前best得分-63.97，平均长度963.75步，terminated仅1/20；landing_contact_success激活率0.0%（死组件），soft_landing与orientation_penalty为全覆盖惩罚，goal_distance_progress per-step仅0.00392。agent在绝大多数回合达到时间上限而悬停，未学到降落行为。",
  "behavior_diagnosis": "策略学到悬停或极慢飘移，避免接近目标垫，因为接近会触发soft_landing速度惩罚，而成功接触奖励从未激活无法提供正导向，最终在时间截断后终止。",
  "signal_completeness": "当前奖励具备距离进步、近端速度惩罚、姿态惩罚和着陆接触奖励职责，但着陆接触奖励为稀疏二元条件且从未触发，缺乏可学习的成功信号，导致着陆动力缺失。",
  "selected_level": "Level 2（修改组件公式形式）。landing_contact_success为死组件，需要将稀疏奖励转化为连续可导形式以提供梯度。",
  "selected_intervention": "唯一目标组件为landing_contact_success。修改为：当双腿接触时给予与速度成反比的连续奖励 `w_contact * max(0, 1 - speed / v_thresh)`，保持低速越高奖励的逻辑，并提升权重w_contact从2.0到5.0。",
  "falsifiable_hypothesis": "连续的接触奖励将给着陆动作提供正梯度，agent会在接近时减速以获取更高奖励，从而提升terminated成功次数和score，dead component激活率上升。",
  "expected_next_round": "landing_contact_success active_rate >0%，terminated数上升（预计≥3），score提升至-30~0区间，回合长度可能先下降后稳定。",
  "main_risk": "接触奖励权重过大可能诱使agent高速砸垫，但soft_landing和orientation_penalty仍会抑制高速，实际风险为agent过早接触而未充分减速导致crash终止增加，可能需要后续调整soft_landing强度。"
}
```

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    prev_x, prev_y = obs[0], obs[1]
    prev_vx, prev_vy = obs[2], obs[3]

    next_x, next_y = next_obs[0], next_obs[1]
    next_vx, next_vy = next_obs[2], next_obs[3]
    next_angle = next_obs[4]
    next_left_contact = next_obs[6]
    next_right_contact = next_obs[7]

    # Distance to target pad (0,0)
    dist_prev = (prev_x**2 + prev_y**2) ** 0.5
    dist_next = (next_x**2 + next_y**2) ** 0.5

    # 1. Goal distance progress (improvement delta)
    w_progress = 5.0
    progress = w_progress * (dist_prev - dist_next)

    # 2. Soft landing (proximity-gated velocity penalty)
    w_vel = 1.0
    k_dist = 20.0
    proximity_gate = 1.0 / (1.0 + k_dist * dist_next**2)
    vel_penalty = -w_vel * (next_vx**2 + next_vy**2) * proximity_gate

    # 3. Orientation penalty (quadratic on tilt)
    w_angle = 0.5
    angle_penalty = -w_angle * (next_angle**2)

    # 4. Landing contact success (continuous, speed-dependent)
    w_contact = 5.0
    v_thresh = 0.5
    legs_on_ground = (next_left_contact >= 0.5) and (next_right_contact >= 0.5)
    if legs_on_ground:
        speed = (next_vx**2 + next_vy**2) ** 0.5
        contact_bonus = w_contact * max(0.0, 1.0 - speed / v_thresh)
    else:
        contact_bonus = 0.0

    # Aggregate
    total_reward = progress + vel_penalty + angle_penalty + contact_bonus

    components = {
        "goal_distance_progress": progress,
        "soft_landing": vel_penalty,
        "orientation_penalty": angle_penalty,
        "landing_contact_success": contact_bonus
    }

    return float(total_reward), components
```